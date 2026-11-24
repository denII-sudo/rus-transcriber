# Хук Claude Code (UserPromptSubmit и Stop): предупреждает, что контекст сессии пора сжать.
# Заполнение считается по транскрипту: токены входа последнего ответа модели, как в строке
# состояния. С порога RT_CONTEXT_WARN_PCT (35%) хук показывает предупреждение человеку, а при
# отправке сообщения просит модель предложить /compact; повторяет один раз на 5% выше порога
# (40%) и дальше молчит до сжатия.
# Само сжатие хук не запускает: автосжатие задаёт CLAUDE_AUTOCOMPACT_PCT_OVERRIDE в
# .claude/settings.json. Размер окна — RT_CONTEXT_WINDOW_TOKENS (по умолчанию 1 000 000).
# Любая ошибка — тишина и код 0: хук не должен мешать работе.
$ErrorActionPreference = "Stop"

function Get-EnvInt([string]$name, [int64]$default, [int64]$min, [int64]$max) {
    $value = [Environment]::GetEnvironmentVariable($name)
    if ($value -match '^\s*\d{1,12}\s*$') {
        $number = [int64]$value
        if ($number -ge $min -and $number -le $max) { return $number }
    }
    return $default
}

# Последний ответ основной сессии после последнего сжатия: число токенов или $null.
function Get-ContextTokens([string]$path) {
    $stream = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read,
        [IO.FileShare]::ReadWrite -bor [IO.FileShare]::Delete)
    try {
        foreach ($size in @(8MB, 64MB)) {
            $take = [Math]::Min($stream.Length, [int64]$size)
            $buffer = New-Object byte[] $take
            [void]$stream.Seek(-$take, [IO.SeekOrigin]::End)
            $read = 0
            while ($read -lt $take) {
                $n = $stream.Read($buffer, $read, $take - $read)
                if ($n -le 0) { break }
                $read += $n
            }
            $lines = [Text.Encoding]::UTF8.GetString($buffer, 0, $read) -split "`n"
            # Первая строка куска может быть обрезана; при чтении всего файла она целая.
            $first = if ($take -eq $stream.Length) { 0 } else { 1 }
            for ($i = $lines.Count - 1; $i -ge $first; $i--) {
                $line = $lines[$i]
                if (-not ($line.Contains('"usage"') -or $line.Contains('"compact_boundary"'))) { continue }
                try { $entry = $line | ConvertFrom-Json } catch { continue }
                if ($entry.type -eq "system" -and $entry.subtype -eq "compact_boundary") { return $null }
                if ($entry.type -ne "assistant" -or $entry.isSidechain -eq $true) { continue }
                $usage = $entry.message.usage
                if ($null -eq $usage -or $entry.message.model -eq "<synthetic>") { continue }
                $total = [int64]$usage.input_tokens + [int64]$usage.cache_read_input_tokens +
                    [int64]$usage.cache_creation_input_tokens
                if ($total -gt 0) { return $total }
            }
            if ($take -eq $stream.Length) { break }
        }
    } finally {
        $stream.Dispose()
    }
    return $null
}

# JSON только из ASCII: кириллица не зависит от кодировки консоли.
function ConvertTo-AsciiJson($value) {
    $json = $value | ConvertTo-Json -Compress -Depth 5
    return [regex]::Replace($json, '[^\x00-\x7F]', { param($m) '\u{0:x4}' -f [int][char]$m.Value })
}

try {
    [Console]::InputEncoding = New-Object Text.UTF8Encoding $false
    $hook = [Console]::In.ReadToEnd() | ConvertFrom-Json
    $hookEvent = [string]$hook.hook_event_name
    $path = [string]$hook.transcript_path
    if ($hookEvent -notin @("UserPromptSubmit", "Stop")) { exit 0 }
    if (-not $path -or -not (Test-Path -LiteralPath $path -PathType Leaf)) { exit 0 }

    $window = Get-EnvInt "RT_CONTEXT_WINDOW_TOKENS" 1000000 100000 100000000
    $warn = Get-EnvInt "RT_CONTEXT_WARN_PCT" 35 1 100
    $step = 5
    $tokens = Get-ContextTokens $path

    $session = ([string]$hook.session_id) -replace '[^A-Za-z0-9-]', ''
    if (-not $session) { $session = "unknown" }
    $stateDir = Join-Path ([IO.Path]::GetTempPath()) "rus-transcriber-context-warning"
    $state = Join-Path $stateDir "$session-$hookEvent.txt"

    $percent = if ($null -eq $tokens) { 0 } else { [int64][Math]::Floor($tokens * 100.0 / $window) }
    if ($percent -lt $warn) {
        # После сжатия заполнение падает: следующий рост предупреждается заново.
        if (Test-Path -LiteralPath $state) { Remove-Item -LiteralPath $state -Force }
        exit 0
    }
    # Два предупреждения: на пороге и на 5% выше; выше 35% — тишина до сжатия.
    $level = [Math]::Min($warn + [Math]::Floor(($percent - $warn) / $step) * $step, $warn + $step)
    $previous = 0
    if (Test-Path -LiteralPath $state) {
        $saved = (Get-Content -LiteralPath $state -Raw).Trim()
        if ($saved -match '^\d+$') { $previous = [int64]$saved }
    }
    if ($level -le $previous) { exit 0 }
    [void](New-Item -ItemType Directory -Force -Path $stateDir)
    Set-Content -LiteralPath $state -Value $level -Encoding ascii

    $used = [Math]::Round($tokens / 1000)
    $auto = Get-EnvInt "CLAUDE_AUTOCOMPACT_PCT_OVERRIDE" 0 1 100
    $tail = if ($auto -gt 0) { " Автосжатие сработает около $auto%." } else { "" }
    $text = "Контекст сессии заполнен на $percent% ($used тыс. токенов). " +
        "Сожмите сессию командой /compact между задачами.$tail"
    $output = @{ systemMessage = $text }
    if ($hookEvent -eq "UserPromptSubmit") {
        $output.hookSpecificOutput = @{
            hookEventName = "UserPromptSubmit"
            additionalContext = "Контекст этой сессии заполнен на $percent%. Одной строкой в начале " +
                "ответа предложи человеку сжать сессию командой /compact, когда текущая задача закончится."
        }
    }
    [Console]::Out.Write((ConvertTo-AsciiJson $output))
    exit 0
} catch {
    exit 0
}
