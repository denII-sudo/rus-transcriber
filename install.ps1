<#
Установка rus-transcriber для Intel Core Ultra / Windows 11.

  powershell -ExecutionPolicy Bypass -File .\install.ps1
      ставит в %LOCALAPPDATA%\rus-transcriber (личная папка, не синхронизируется с OneDrive)

  powershell -ExecutionPolicy Bypass -File .\install.ps1 -Target D:\rus-transcriber
      ставит в другую папку и закрывает к ней доступ всем, кроме вас, SYSTEM и администраторов
      (нужно, если в имени пользователя Windows есть кириллица или мало места на C:)

  powershell -ExecutionPolicy Bypass -File .\install.ps1 -DataDir D:\rus-transcriber-data
      папка личных данных (голосовые профили, словари групп, настройки, задачи): по умолчанию
      %APPDATA%\rus-transcriber-data; путь записывается в настройки сервера (RUS_TRANSCRIBER_DATA)

  Обновление уже стоящей программы — тот же запуск из папки новой версии. Перед изменениями
  установщик показывает версии и коммит источника, проверяет, что программа не занята, и
  сохраняет список пакетов (cache\pip-freeze) для отката; в конце запускает самопроверку.
      -KeepSavedToken        не спрашивать токен Hugging Face: оставить сохранённый в настройках
                             (без терминала вопрос не задаётся и так)
      -AllowRunningServer    обновлять, пока работает только сервер программы (его запустило
                             приложение ИИ-агента); идущая обработка записи всё равно останавливает
      -AllowUnreviewedSource ставить из клона с незакоммиченными изменениями или из коммита,
                             которого нет в origin/main и на котором нет тега
      -SkipSelfCheck         не запускать самопроверку в конце

  powershell -ExecutionPolicy Bypass -File .\install.ps1 -AcceptanceCopyOnly -Target <sandbox>\program
      -DataDir <sandbox>\data -SkipClaudeConfig
      только копирование runtime-файлов для локальной приёмки; требует
      RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX=1, а обе папки должны лежать внутри одной изолированной
      папки с файлом-меткой rus-transcriber-acceptance.sandbox (её создаёт тест приёмки во временной
      папке). Настоящие папки программы и данных, а также ссылки (junction/symlink) не принимаются.

  powershell -ExecutionPolicy Bypass -File .\install.ps1 -Uninstall
      сначала проверяет, что программа не работает, и показывает, что будет удалено;
      затем спрашивает: удалить программу и убрать её из настроек приложения ИИ-агента
      (по умолчанию да), удалить служебные данные (по умолчанию НЕТ), удалить резервные
      копии настроек с токеном (по умолчанию НЕТ). Аудиозаписи, стенограммы, конспекты и
      база state.sqlite3 не удаляются никогда.
      Без вопросов (для ИИ-агента): -Yes и -RemoveData или -KeepData, по желанию -RemoveConfigBackups

Повторный запуск безопасен: обновляет программу, не трогая данные пользователя (папка данных: профили, словари, config.json с результатом теста скорости).
#>
param(
    [string]$Target = "",
    [string]$DataDir = "",
    [string]$Token = "",
    [switch]$SkipClaudeConfig,
    [switch]$AcceptanceCopyOnly,
    [switch]$Uninstall,
    [switch]$Yes,
    [switch]$RemoveData,
    [switch]$KeepData,
    [switch]$RemoveConfigBackups,
    [switch]$KeepSavedToken,
    [switch]$AllowRunningServer,
    [switch]$AllowUnreviewedSource,
    [switch]$SkipSelfCheck
)
$ErrorActionPreference = "Continue"
# Windows PowerShell 5.1: без этого ConvertTo-Json пишет массивы как {"value":[...],"Count":n}
Remove-TypeData System.Array -ErrorAction SilentlyContinue
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Source = $PSScriptRoot
$Name = "rus-transcriber"

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail "Ошибка на шаге: $what (код $LASTEXITCODE)" } }
function Check-Pip($what) {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Если выше есть CERTIFICATE_VERIFY_FAILED / certificate has expired, см. инструкцию: $(Join-Path $Target 'docs\model-download-certificate.md')" -ForegroundColor Yellow
        Fail "Ошибка установки пакета: $what (код $LASTEXITCODE)"
    }
}
function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function HasNonAscii($s) { return $s -match '[^\x00-\x7F]' }
function WriteUtf8NoBom($path, $text) { [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false)) }

if ($AcceptanceCopyOnly) {
    if ($Uninstall) { Fail "-AcceptanceCopyOnly нельзя сочетать с -Uninstall." }
    if (-not $Target -or -not $DataDir -or -not $SkipClaudeConfig) {
        Fail "-AcceptanceCopyOnly требует явные -Target, -DataDir и -SkipClaudeConfig."
    }
    if ($env:RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX -ne "1") {
        Fail "Для режима приёмки задайте RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX=1."
    }
}
if ($Token -and $KeepSavedToken) { Fail "Укажите что-то одно: -Token или -KeepSavedToken." }
# без терминала (ИИ-агент, перенаправленный ввод) вопросы Read-Host не задаются
$nonInteractive = [Console]::IsInputRedirected

# ---------- помощники обновления (без побочных эффектов; проверяются tests/install_update_harness.ps1) ----------
# Процессы программы из папки $root, кроме самого установщика и его родителей.
function Get-ProgramProcesses([string]$root) {
    $chain = @($PID)
    $cur = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction SilentlyContinue
    while ($cur -and $cur.ParentProcessId -and $chain.Count -lt 10) {
        $chain += [int]$cur.ParentProcessId
        $cur = Get-CimInstance Win32_Process -Filter "ProcessId=$($cur.ParentProcessId)" -ErrorAction SilentlyContinue
    }
    $prefix = $root + [IO.Path]::DirectorySeparatorChar
    return @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
        ($chain -notcontains [int]$_.ProcessId) -and (
            ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) -or
            ($_.Name -match '^(python|pythonw|opencode)' -and $_.CommandLine -and
             $_.CommandLine.IndexOf($root, [StringComparison]::OrdinalIgnoreCase) -ge 0))
    })
}
# Делит процессы программы на задачи (расшифровка, обучение голосов, тест скорости) и остальное (сервер и др.).
function Split-ProgramProcesses($procs) {
    $jobs = @(); $other = @()
    foreach ($p in @($procs)) {
        if (-not $p) { continue }      # пустой результат PowerShell передаёт как $null
        if ($p.CommandLine -and $p.CommandLine -match '(worker|bench)\.py') { $jobs += $p } else { $other += $p }
    }
    return @{ jobs = $jobs; other = $other }
}
# Версия программы (VERSION в server.py) в папке $dir или пустая строка.
function Get-ServerVersion([string]$dir) {
    $f = Join-Path $dir "server.py"
    if (-not (Test-Path -LiteralPath $f)) { return "" }
    $m = Select-String -LiteralPath $f -Pattern '^VERSION\s*=\s*"([^"]+)"' -Encoding UTF8 | Select-Object -First 1
    if ($m) { return $m.Matches[0].Groups[1].Value }
    return ""
}
# Состояние источника установки: Git-клон или папка без Git. Сеть не используется:
# сравнение с origin/main — по последнему git fetch.
function Get-SourceState([string]$dir) {
    $s = [ordered]@{ git = $false; gitTool = $false; commit = ""; branch = ""; dirty = @(); tag = ""; onMain = $false; reviewed = $true }
    if (-not (Test-Path -LiteralPath (Join-Path $dir ".git"))) { return $s }
    $s.git = $true
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) { $s.reviewed = $false; return $s }
    $s.gitTool = $true
    $s.commit = [string](& git -C $dir rev-parse --short HEAD 2>$null)
    $s.branch = [string](& git -C $dir rev-parse --abbrev-ref HEAD 2>$null)
    $s.dirty = @(& git -C $dir status --porcelain 2>$null | Where-Object { $_ })
    $s.tag = [string](& git -C $dir describe --tags --exact-match HEAD 2>$null)
    & git -C $dir merge-base --is-ancestor HEAD origin/main 2>$null
    $s.onMain = ($LASTEXITCODE -eq 0)
    $s.reviewed = ($s.commit -and -not $s.dirty.Count -and ($s.onMain -or $s.tag))
    return $s
}
# Что сделать с onnxruntime по установленным версиям: обычные сборки (onnxruntime, -directml, -gpu)
# и onnxruntime-openvino дают один модуль; переставлять сборку с OpenVINO — только если нужно.
function Get-OnnxPlan($versions, [string]$wanted) {
    $remove = @(@("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu") | Where-Object { $versions[$_] })
    # удаление обычной сборки стирает общий модуль — после него сборку с OpenVINO ставим заново
    $reinstall = ($remove.Count -gt 0) -or ($versions["onnxruntime-openvino"] -ne $wanted)
    return @{ remove = $remove; reinstall = $reinstall }
}
# Как поступить с токеном Hugging Face: given (передан -Token), saved (оставить сохранённый),
# none (пропустить), ask (спросить человека).
function Get-TokenMode([string]$token, [bool]$keepSaved, [string]$saved, [bool]$noTerminal) {
    if ($token) { return "given" }
    if ($keepSaved -or $noTerminal) { if ($saved) { return "saved" } else { return "none" } }
    return "ask"
}
# Разница двух списков pip freeze: какие пакеты добавлены, удалены или сменили версию.
function Compare-PipFreeze([string[]]$before, [string[]]$after) {
    function ToMap($lines) {
        $m = @{}
        foreach ($l in @($lines)) {
            $l = ([string]$l).Trim()
            if (-not $l -or $l.StartsWith("#")) { continue }
            $n = ($l -split '(==| @ )', 2)[0].Trim().ToLowerInvariant().Replace("_", "-")
            $m[$n] = $l
        }
        return $m
    }
    $b = ToMap $before; $a = ToMap $after; $changes = @()
    foreach ($k in (@($b.Keys) + @($a.Keys) | Sort-Object -Unique)) {
        if (-not $a.ContainsKey($k)) { $changes += "удалён: $($b[$k])" }
        elseif (-not $b.ContainsKey($k)) { $changes += "добавлен: $($a[$k])" }
        elseif ($a[$k] -ne $b[$k]) { $changes += "изменён: $($b[$k]) -> $($a[$k])" }
    }
    return ,$changes
}

# Добавляет в PATH текущего окна пути, появившиеся после установки winget-пакетов
# (реестр Machine/User и папка ссылок на portable-пакеты). Позволяет не требовать
# перезапуска PowerShell посреди установки.
function Update-SessionPath {
    $current = $env:PATH
    $existing = @($current -split ";" | Where-Object { $_ })
    $extra = @()
    foreach ($scope in @("Machine", "User")) {
        $p = [Environment]::GetEnvironmentVariable("Path", $scope)
        if ($p) { $extra += ($p -split ";" | Where-Object { $_.Trim() } | ForEach-Object { $_.Trim() }) }
    }
    $links = Join-Path $env:LOCALAPPDATA "Microsoft\WinGet\Links"
    if (Test-Path $links) { $extra += $links }
    foreach ($x in $extra) {
        if ($existing -notcontains $x) { $current = $current.TrimEnd(";") + ";" + $x; $existing += $x }
    }
    $env:PATH = $current
}
function Invoke-RerunHint {
    return "  powershell -ExecutionPolicy Bypass -File `"$PSCommandPath`""
}

# ---------- где лежит конфиг приложения ИИ-агента (обычная установка или из Microsoft Store) ----------
function Find-ClaudeConfig {
    # Claude Desktop из Microsoft Store (с версии ~3.1623) читает настройки из
    #   %LOCALAPPDATA%\Claude-Data\claude_<id>\claude_desktop_config.json
    # Обычная установка — из %APPDATA%\Claude\claude_desktop_config.json. Пишем во все существующие места.
    $result = @()
    $dataRoot = Join-Path $env:LOCALAPPDATA "Claude-Data"
    $msix = Get-ChildItem $dataRoot -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match '^claude_[a-z0-9]+$' }
    foreach ($d in $msix) { $result += (Join-Path $d.FullName "claude_desktop_config.json") }
    $std = Join-Path $env:APPDATA "Claude\claude_desktop_config.json"
    if ((Test-Path $std) -or -not $result) { $result += $std }
    return @($result | Select-Object -Unique)
}

# прошлая версия установщика могла по ошибке создать файл в корне Claude-Data — убираем, если там только наша запись
# (в режиме приёмки настоящий профиль не трогаем)
$stray = Join-Path $env:LOCALAPPDATA "Claude-Data\claude_desktop_config.json"
if (-not $AcceptanceCopyOnly -and (Test-Path $stray)) {
    try {
        $j = Get-Content $stray -Raw -Encoding UTF8 | ConvertFrom-Json
        $keys = @($j.PSObject.Properties.Name)
        if ($keys.Count -eq 1 -and $keys[0] -eq "mcpServers") { Remove-Item $stray -Force; Remove-Item "$stray.bak-*" -Force -ErrorAction SilentlyContinue }
    } catch { }
}

function Edit-ClaudeConfig($path, [scriptblock]$change) {
    $dir = Split-Path $path
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    $cfg = [pscustomobject]@{}
    if (Test-Path $path) {
        $raw = Get-Content $path -Raw -Encoding UTF8
        if ($raw -and $raw.Trim()) {
            try { $cfg = $raw | ConvertFrom-Json } catch { Fail "Не читается $path (ошибка в JSON). Исправьте файл или запустите с -SkipClaudeConfig." }
        }
        $backup = "$path.bak-" + (Get-Date -Format "yyyyMMdd-HHmmss")
        Copy-Item $path $backup
        Write-Host "  резервная копия настроек: $backup" -ForegroundColor DarkGray
    }
    if (-not ($cfg.PSObject.Properties.Name -contains "mcpServers")) {
        $cfg | Add-Member -NotePropertyName mcpServers -NotePropertyValue ([pscustomobject]@{})
    }
    & $change $cfg.mcpServers
    WriteUtf8NoBom $path ($cfg | ConvertTo-Json -Depth 32)
}

# ---------- поиск существующей установки ----------
if (-not $Target) {
    $Target = Join-Path $env:LOCALAPPDATA $Name
    foreach ($cp in (Find-ClaudeConfig)) {
        if (Test-Path $cp) {
            try {
                $c = Get-Content $cp -Raw -Encoding UTF8 | ConvertFrom-Json
                $srv = $c.mcpServers.$Name
                if ($srv -and $srv.args) { $Target = Split-Path ([string]$srv.args[0]) }
                if ($srv -and $srv.env -and $srv.env.RUS_TRANSCRIBER_DATA -and -not $DataDir) { $DataDir = [string]$srv.env.RUS_TRANSCRIBER_DATA }
            } catch { }
        }
    }
}
$Target = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Target).TrimEnd("\")
# папка данных: параметр -DataDir -> уже записанная в настройках приложения ИИ-агента -> %APPDATA%\rus-transcriber-data
if (-not $DataDir) { $DataDir = Join-Path $env:APPDATA "$Name-data" }
$DataDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataDir).TrimEnd("\")
if ($DataDir -eq $Target -or $DataDir.StartsWith($Target + "\", [StringComparison]::OrdinalIgnoreCase)) {
    Fail "Папка данных ($DataDir) не может лежать внутри папки программы ($Target): при обновлении она бы потерялась. Укажите другую через -DataDir."
}
if ($AcceptanceCopyOnly) {
    # Режим приёмки работает только в изолированной папке, помеченной файлом-меткой. Фиксированного
    # пути нет: такая папка создаётся во временной папке на любом компьютере (тест, self_check, CI).
    $markerName = "rus-transcriber-acceptance.sandbox"
    $Target = [IO.Path]::GetFullPath($Target).TrimEnd("\")
    $DataDir = [IO.Path]::GetFullPath($DataDir).TrimEnd("\")
    function Test-Inside([string]$path, [string]$root) {
        return $path.StartsWith($root.TrimEnd("\") + "\", [StringComparison]::OrdinalIgnoreCase)
    }
    function Test-Overlap([string]$a, [string]$b) {
        return ($a -ieq $b) -or (Test-Inside $a $b) -or (Test-Inside $b $a)
    }
    # ближайшая родительская папка Target, в которой лежит метка
    $sandboxRoot = $null
    $dir = Split-Path $Target -Parent
    while ($dir) {
        if (Test-Path -LiteralPath (Join-Path $dir $markerName)) { $sandboxRoot = $dir.TrimEnd("\"); break }
        $parent = Split-Path $dir -Parent
        if (-not $parent -or $parent -eq $dir) { break }
        $dir = $parent
    }
    if (-not $sandboxRoot) {
        Fail "-AcceptanceCopyOnly разрешён только внутри изолированной папки с файлом-меткой $markerName (её создаёт тест приёмки)."
    }
    $markerItem = Get-Item -LiteralPath (Join-Path $sandboxRoot $markerName) -Force
    if ($markerItem.PSIsContainer -or ($markerItem.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
        Fail "Метка $markerName должна быть обычным файлом: $($markerItem.FullName)"
    }
    if (-not (Test-Inside $DataDir $sandboxRoot)) {
        Fail "-AcceptanceCopyOnly: папка данных должна лежать в той же изолированной папке ($sandboxRoot)."
    }
    # ни одна существующая папка на пути от метки до Target и DataDir не может быть ссылкой:
    # иначе копирование ушло бы за пределы изолированной папки
    foreach ($path in @($Target, $DataDir)) {
        $cur = $path
        while ($true) {
            if (Test-Path -LiteralPath $cur) {
                if ((Get-Item -LiteralPath $cur -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) {
                    Fail "-AcceptanceCopyOnly не принимает ссылки/reparse-папки: $cur"
                }
            }
            if ($cur -ieq $sandboxRoot) { break }
            $cur = Split-Path $cur -Parent
        }
    }
    # настоящая установка и данные не годятся для приёмки, даже если рядом оказалась метка
    $protected = @((Join-Path $env:LOCALAPPDATA $Name), (Join-Path $env:APPDATA "$Name-data"), $Source)
    foreach ($cp in (Find-ClaudeConfig)) {
        if (Test-Path $cp) {
            try {
                $srv = (Get-Content $cp -Raw -Encoding UTF8 | ConvertFrom-Json).mcpServers.$Name
                if ($srv -and $srv.args) { $protected += (Split-Path ([string]$srv.args[0])) }
                if ($srv -and $srv.env -and $srv.env.RUS_TRANSCRIBER_DATA) { $protected += [string]$srv.env.RUS_TRANSCRIBER_DATA }
            } catch { }
        }
    }
    foreach ($p in $protected) {
        if (-not $p) { continue }
        $full = [IO.Path]::GetFullPath($p).TrimEnd("\")
        foreach ($path in @($Target, $DataDir)) {
            if (Test-Overlap $path $full) { Fail "-AcceptanceCopyOnly не работает с настоящей папкой программы, данных или исходников: $full" }
        }
    }
    foreach ($bad in @(".git", ".venv")) {
        if (Test-Path -LiteralPath (Join-Path $Target $bad)) { Fail "-AcceptanceCopyOnly: в $Target уже есть $bad — это не изолированная папка приёмки." }
    }
    if ($Target.StartsWith($DataDir + "\", [StringComparison]::OrdinalIgnoreCase) -or
        $DataDir.StartsWith($Target + "\", [StringComparison]::OrdinalIgnoreCase)) {
        Fail "Папки program и data в режиме приёмки должны быть раздельными."
    }
}

# ================================ УДАЛЕНИЕ ================================
if ($Uninstall) {
    # --- запуск из папки программы: продолжить из временной копии, чтобы не удалять работающий скрипт ---
    if ($PSCommandPath -and $PSCommandPath.StartsWith($Target + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
        $copy = Join-Path $env:TEMP ("$Name-uninstall-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".ps1")
        Copy-Item $PSCommandPath $copy -Force
        $argsList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $copy, "-Uninstall",
                      "-Target", $Target, "-DataDir", $DataDir)
        if ($Yes) { $argsList += "-Yes" }
        if ($RemoveData) { $argsList += "-RemoveData" }
        if ($KeepData) { $argsList += "-KeepData" }
        if ($RemoveConfigBackups) { $argsList += "-RemoveConfigBackups" }
        $shell = (Get-Process -Id $PID).Path
        & $shell @argsList
        exit $LASTEXITCODE
    }
    if ($RemoveData -and $KeepData) { Fail "Укажите что-то одно: -RemoveData или -KeepData." }
    # Папка программы с .git — это рабочая копия разработки: удаление стёрло бы незакоммиченную работу.
    if (Test-Path -LiteralPath (Join-Path $Target ".git")) {
        Fail ("В папке программы есть .git: $Target — это рабочая копия разработки, а не только установка.`n" +
              "Удаление не начато, ничего не изменено. Сначала перенесите разработку в отдельный клон " +
              "и убедитесь, что вся работа закоммичена и отправлена; затем уберите .git из папки программы.")
    }
    # Аудиозаписи, стенограммы, конспекты и базы с ними не удаляются никогда, даже внутри папок программы и данных.
    $AudioExt = @(".mp3", ".wav", ".m4a", ".ogg", ".oga", ".opus", ".flac", ".aac", ".wma", ".webm",
                  ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".amr", ".3gp")
    function Test-UserItem($item, [string[]]$extraNames) {
        $n = $item.Name.ToLowerInvariant()
        if ($item.PSIsContainer) {
            if ($n.EndsWith(".rus-transcriber") -or $n.EndsWith(".review")) { return $true }
        } else {
            if ($AudioExt -contains $item.Extension.ToLowerInvariant()) { return $true }
            foreach ($suffix in @(".transcript.md", ".transcript.json", ".conspect.md", ".conspect.json",
                                  ".sqlite3", ".sqlite3-wal", ".sqlite3-shm")) {
                if ($n.EndsWith($suffix)) { return $true }
            }
        }
        foreach ($pattern in $extraNames) { if ($n -like $pattern) { return $true } }
        return $false
    }
    # Ищет файлы пользователя под $root, не заходя в ссылки (junction/symlink) и в $skipTop (окружение, кэши).
    function Find-UserItems($root, [string[]]$skipTop, [string[]]$extraTop) {
        $found = New-Object System.Collections.ArrayList
        $stack = New-Object System.Collections.Stack
        $stack.Push($root)
        while ($stack.Count) {
            $dir = $stack.Pop()
            foreach ($item in @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue)) {
                $top = ($dir -eq $root)
                if ($top -and ($skipTop -contains $item.Name)) { continue }
                $extra = @(); if ($top) { $extra = $extraTop }
                if (Test-UserItem $item $extra) { [void]$found.Add($item.FullName); continue }
                if ($item.PSIsContainer -and -not ($item.Attributes -band [IO.FileAttributes]::ReparsePoint)) {
                    $stack.Push($item.FullName)
                }
            }
        }
        return ,$found
    }
    # Удаляет дерево, не переходя по ссылкам: ссылка удаляется сама, её цель не трогается.
    function Remove-Tree($path) {
        $item = Get-Item -LiteralPath $path -Force -ErrorAction SilentlyContinue
        if (-not $item) { return }
        if (-not $item.PSIsContainer) {
            $item.Attributes = $item.Attributes -band (-bnot [IO.FileAttributes]::ReadOnly)
            [IO.File]::Delete($item.FullName); return
        }
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) { [IO.Directory]::Delete($item.FullName); return }
        try { [IO.Directory]::Delete($item.FullName, $true); return } catch { }
        foreach ($child in @(Get-ChildItem -LiteralPath $item.FullName -Force)) { Remove-Tree $child.FullName }
        [IO.Directory]::Delete($item.FullName)
    }
    # Удаляет содержимое $root, кроме найденных файлов пользователя; возвращает список сохранённого.
    function Remove-ExceptUser($root, [string[]]$skipTop, [string[]]$extraTop) {
        $keep = Find-UserItems $root $skipTop $extraTop
        $failed = New-Object System.Collections.ArrayList
        function Sweep($dir) {
            foreach ($item in @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue)) {
                $p = $item.FullName
                if ($keep -contains $p) { continue }
                $holdsUser = $false
                foreach ($k in $keep) { if ($k.StartsWith($p + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) { $holdsUser = $true; break } }
                if ($holdsUser) { Sweep $p; continue }
                try { Remove-Tree $p } catch { [void]$failed.Add($p) }
            }
            if ($dir -ne $root -and -not @(Get-ChildItem -LiteralPath $dir -Force -ErrorAction SilentlyContinue)) {
                try { [IO.Directory]::Delete($dir) } catch { }
            }
        }
        Sweep $root
        if (-not $keep.Count -and -not $failed.Count) { try { [IO.Directory]::Delete($root) } catch { } }
        return @{ kept = $keep; failed = $failed }
    }
    function Ask-YesNo($question, [bool]$defaultYes) {
        $hint = "(y/N)"; if ($defaultYes) { $hint = "(Y/n)" }
        $ans = Read-Host "$question $hint"
        if (-not $ans) { return $defaultYes }
        return ($ans -match '^(y|д)')
    }

    $programFound = Test-Path (Join-Path $Target "server.py")
    $configs = @(Find-ClaudeConfig | Where-Object { Test-Path $_ })
    $withEntry = @(); $badConfigs = @()
    foreach ($cp in $configs) {
        try {
            $c = Get-Content $cp -Raw -Encoding UTF8 | ConvertFrom-Json
            if ($c.mcpServers -and ($c.mcpServers.PSObject.Properties.Name -contains $Name)) { $withEntry += $cp }
        } catch { $badConfigs += $cp }
    }
    if ($badConfigs) {
        Fail ("Не читается файл настроек ИИ-агента (ошибка в JSON): $($badConfigs -join ', ')`n" +
              "Исправьте его или восстановите из резервной копии рядом (*.bak-*). Ничего не изменено.")
    }
    $backupsBefore = @($configs | ForEach-Object { Get-ChildItem "$_.bak-*" -ErrorAction SilentlyContinue })

    Write-Host "Удаление $Name" -ForegroundColor Yellow
    # --- 1. программа не должна работать: иначе файлы заняты и удаление пройдёт наполовину ---
    while ($programFound) {
        $running = Get-ProgramProcesses $Target
        if (-not $running) { break }
        Write-Host "Программа сейчас работает (её запустило приложение ИИ-агента или идёт обработка записи):" -ForegroundColor Yellow
        $running | ForEach-Object { Write-Host ("  {0} (PID {1})" -f $_.Name, $_.ProcessId) }
        Write-Host "Полностью закройте приложение ИИ-агента (например, Claude Desktop: значок в трее → Quit) и дождитесь окончания обработки."
        if ($nonInteractive) { Fail "Удаление не начато: программа работает. Ничего не изменено." }
        $ans = Read-Host "Нажмите Enter, когда закроете, или введите q, чтобы выйти без изменений"
        if ($ans -match '^(q|й)') { Write-Host "Удаление отменено, ничего не изменено."; exit 0 }
    }

    # --- 2. что будет сделано ---
    Write-Host ""
    Write-Host "Что будет удалено (с вашего подтверждения):" -ForegroundColor Cyan
    if ($programFound) {
        Write-Host "  • программа: $Target — код, окружение Python, модели и кэши"
        foreach ($cp in $withEntry) { Write-Host "  • запись программы (и токен Hugging Face) в настройках ИИ-агента: $cp" }
    } else {
        Write-Host "  • программа в $Target не найдена — удалять нечего"
    }
    if (Test-Path $DataDir) {
        Write-Host "  • служебные данные программы (спрошу отдельно): $DataDir"
        Write-Host "    голосовые профили и словари групп, настройки и результаты теста скорости, журналы задач"
    }
    if ($backupsBefore) {
        Write-Host "  • резервные копии настроек ИИ-агента, сделанные установщиком (спрошу отдельно): $($backupsBefore.Count) шт.; в них есть токен Hugging Face"
    }
    Write-Host "Что не удаляется никогда:" -ForegroundColor Cyan
    Write-Host "  • ваши аудиозаписи, стенограммы и конспекты — где бы они ни лежали, в том числе внутри папок программы и данных"
    Write-Host "  • база state.sqlite3 в папке данных (в ней хранятся стенограммы и конспекты), резервные копии и архивы выгрузки"
    Write-Host "  • папки backup* и _backup* внутри папки программы"
    Write-Host "  • Python и ffmpeg (ими могут пользоваться другие программы)"
    Write-Host ""

    $removed = @(); $kept = @(); $failed = @()
    # --- 3. программа (по Enter — да) ---
    if ($programFound) {
        if ($Yes) { $doProgram = $true }
        elseif ($nonInteractive) { Fail "Нужно подтверждение. Для удаления без вопросов: -Yes, данные: -RemoveData или -KeepData. Ничего не изменено." }
        else { $doProgram = Ask-YesNo "Удалить программу и убрать её из настроек ИИ-агента?" $true }
        if ($doProgram) {
            foreach ($cp in $withEntry) {
                Edit-ClaudeConfig $cp { param($s) $s.PSObject.Properties.Remove($Name) }
                $removed += "запись в настройках ИИ-агента: $cp"
            }
            # папки резервных копий (backup*, _backup*) в папке программы — тоже не удаляем
        $r = Remove-ExceptUser $Target @(".venv", "cache") @("backup*", "_backup*")
            $kept += $r.kept; $failed += $r.failed
            $removed += "программа: $Target"
        } else { Write-Host "  программа оставлена" }
    }
    # --- 4. служебные данные (по Enter — нет) ---
    if (Test-Path $DataDir) {
        if ($RemoveData) { $doData = $true }
        elseif ($KeepData -or $nonInteractive) { $doData = $false }
        else {
            Write-Host "Служебные данные программы: $DataDir" -ForegroundColor Yellow
            Write-Host "  голосовые профили участников, словари групп, настройки, журналы задач. Аудиозаписи, стенограммы,"
            Write-Host "  конспекты, база state.sqlite3, резервные копии и архивы выгрузки останутся на месте."
            $doData = Ask-YesNo "Удалить служебные данные? Это нельзя отменить." $false
        }
        if ($doData) {
            $r = Remove-ExceptUser $DataDir @() @("exports", "_backup*", "_deleted")
            $kept += $r.kept; $failed += $r.failed
            $removed += "служебные данные: $DataDir"
        } else { Write-Host "  служебные данные оставлены: $DataDir" }
    }
    # --- 5. резервные копии настроек ИИ-агента с токеном (по Enter — нет) ---
    $backups = @($configs | ForEach-Object { Get-ChildItem "$_.bak-*" -ErrorAction SilentlyContinue })
    if ($backups) {
        if ($RemoveConfigBackups) { $doBackups = $true }
        elseif ($nonInteractive) { $doBackups = $false }
        else {
            Write-Host "Резервные копии настроек ИИ-агента ($($backups.Count) шт.) содержат токен Hugging Face и прежние настройки:" -ForegroundColor Yellow
            $backups | Select-Object -First 5 | ForEach-Object { Write-Host "  $($_.FullName)" }
            $doBackups = Ask-YesNo "Удалить эти резервные копии?" $false
        }
        if ($doBackups) {
            foreach ($b in $backups) {
                try { Remove-Item -LiteralPath $b.FullName -Force -ErrorAction Stop; $removed += "резервная копия настроек: $($b.FullName)" }
                catch { $failed += $b.FullName }
            }
        } else { Write-Host "  резервные копии настроек оставлены" }
    }

    # --- 6. итог ---
    Write-Host ""
    Write-Host "Итог удаления" -ForegroundColor Cyan
    if ($removed) { $removed | ForEach-Object { Write-Host "  удалено: $_" } } else { Write-Host "  ничего не удалено" }
    if ($kept) {
        Write-Host "  сохранены ваши файлы ($($kept.Count)), папки вокруг них оставлены:" -ForegroundColor Green
        $kept | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" }
        if ($kept.Count -gt 20) { Write-Host "    … и ещё $($kept.Count - 20)" }
    }
    if ($failed) {
        Write-Host "  не удалось удалить ($($failed.Count)) — файлы заняты или нет прав; удалите вручную после перезагрузки:" -ForegroundColor Red
        $failed | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" }
    }
    Write-Host "  не затрагивались: аудиозаписи и результаты рядом с ними, Python, ffmpeg"
    if ($removed -match "настройках ИИ-агента") { Write-Host "Перезапустите приложение ИИ-агента (например, Claude Desktop)." }
    $tempPrefix = Join-Path $env:TEMP "$Name-uninstall-"
    if ($PSCommandPath -and $PSCommandPath.StartsWith($tempPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $PSCommandPath -Force -ErrorAction SilentlyContinue
    }
    if ($failed) { exit 1 }
    exit 0
}
# ================================ УСТАНОВКА ================================
Write-Host "Установка $Name в: $Target" -ForegroundColor Green
# Установка поверх чужой рабочей копии Git смешала бы программу и разработку (и перезаписала бы файлы клона).
if ($Source -ne $Target -and (Test-Path -LiteralPath (Join-Path $Target ".git"))) {
    Fail ("В папке установки есть .git: $Target — это рабочая копия разработки.`n" +
          "Установка не начата. Держите разработку в отдельном клоне, а программу ставьте в папку без .git " +
          "(подробности: docs/development-and-installation.md).")
}

# ---------- перед изменениями: что ставится и не занята ли программа ----------
$isUpdate = Test-Path -LiteralPath (Join-Path $Target "server.py")
if (-not $AcceptanceCopyOnly) {
    $srcVer = Get-ServerVersion $Source
    if ($Source -eq $Target) { $from = "запуск из папки программы" }
    elseif ($isUpdate) { $from = "сейчас установлена " + $(if ($oldVer = Get-ServerVersion $Target) { $oldVer } else { "версия неизвестна" }) }
    else { $from = "новая установка" }
    Write-Host "  версия программы: $srcVer ($from)"
    $src = Get-SourceState $Source
    if (-not $src.git) {
        Write-Host "  источник: папка без Git (архив или копия): $Source"
    } elseif (-not $src.gitTool) {
        Write-Host "  источник — клон Git, но команда git не найдена: проверить коммит и изменения нельзя" -ForegroundColor Yellow
    } else {
        $where = @()
        if ($src.tag) { $where += "тег $($src.tag)" }
        if ($src.onMain) { $where += "есть в origin/main (по последнему git fetch)" } else { $where += "нет в origin/main" }
        Write-Host "  источник: клон Git $Source, ветка $($src.branch), коммит $($src.commit) — $($where -join ', ')"
        if ($src.dirty.Count) {
            Write-Host "  в клоне есть незакоммиченные изменения ($($src.dirty.Count)):" -ForegroundColor Yellow
            $src.dirty | Select-Object -First 10 | ForEach-Object { Write-Host "    $_" -ForegroundColor Yellow }
        }
    }
    if (-not $src.reviewed) {
        Write-Host "  Этот код не проверен: в main его нет, тега нет или есть незакоммиченные изменения." -ForegroundColor Yellow
        if ($AllowUnreviewedSource) {
            Write-Host "  продолжаю: указан -AllowUnreviewedSource" -ForegroundColor Yellow
        } elseif ($nonInteractive) {
            Fail ("Установка не начата, ничего не изменено. Ставьте из main или тега без незакоммиченных изменений; " +
                  "если этот код ставится сознательно — добавьте -AllowUnreviewedSource.")
        } else {
            $ans = Read-Host "  Всё равно установить? (y/N)"
            if (-not ($ans -match '^(y|д)')) { Write-Host "Установка отменена, ничего не изменено."; exit 0 }
        }
    }
    # Работающая программа держит файлы и библиотеки: обновление пакетов прошло бы наполовину.
    while ($isUpdate) {
        $split = Split-ProgramProcesses (Get-ProgramProcesses $Target)
        if (-not $split.jobs.Count -and -not $split.other.Count) { break }
        if ($split.jobs.Count) {
            Write-Host "Идёт обработка записи или тест скорости:" -ForegroundColor Yellow
            $split.jobs | ForEach-Object { Write-Host ("  {0} (PID {1})" -f $_.Name, $_.ProcessId) }
            Write-Host "Дождитесь окончания или отмените задачу (cancel_job), затем запустите установку снова."
            if ($nonInteractive) { Fail "Обновление не начато: идёт обработка. Ничего не изменено." }
        } else {
            if ($AllowRunningServer) {
                Write-Host "  сервер программы работает — продолжаю (-AllowRunningServer); после обновления перезапустите приложение ИИ-агента" -ForegroundColor Yellow
                break
            }
            Write-Host "Программа сейчас работает (её запустило приложение ИИ-агента):" -ForegroundColor Yellow
            $split.other | ForEach-Object { Write-Host ("  {0} (PID {1})" -f $_.Name, $_.ProcessId) }
            Write-Host "Полностью закройте приложение ИИ-агента (например, Claude Desktop: значок в трее → Quit)."
            if ($nonInteractive) {
                Fail ("Обновление не начато: программа работает. Ничего не изменено. Закройте приложение ИИ-агента " +
                      "или, если обработки нет и можно перезапустить его после обновления, добавьте -AllowRunningServer.")
            }
        }
        $ans = Read-Host "Нажмите Enter, когда закроете, или введите q, чтобы выйти без изменений"
        if ($ans -match '^(q|й)') { Write-Host "Обновление отменено, ничего не изменено."; exit 0 }
    }
}

$insideProfile =$Target.StartsWith($env:USERPROFILE + "\", [StringComparison]::OrdinalIgnoreCase)
if ($AcceptanceCopyOnly) {
    Write-Host "  режим приёмки: проверки места, Python и ffmpeg пропущены"
} else {
Step "1/10" "Проверка места установки"
if (HasNonAscii $Target) {
    Fail ("В пути есть кириллица или другие не-латинские символы: $Target`n" +
          "Часть библиотек (OpenVINO, PyTorch) с такими путями работает ненадёжно.`n" +
          "Запустите с латинским путём, например:  .\install.ps1 -Target C:\rus-transcriber")
}
foreach ($od in @($env:OneDrive, $env:OneDriveConsumer, $env:OneDriveCommercial)) {
    if ($od -and $Target.StartsWith($od, [StringComparison]::OrdinalIgnoreCase)) {
        Fail "Папка внутри OneDrive ($od). Синхронизация сломает окружение Python. Выберите другое место."
    }
}
$drive = (Split-Path -Qualifier $Target) + "\"
$free = (Get-PSDrive ($drive.Substring(0,1))).Free / 1GB
Write-Host ("  свободно на {0} {1:N1} ГБ (нужно около 8)" -f $drive, $free)
if ($free -lt 8) { Fail "Мало места. Освободите диск или укажите -Target на другом диске." }

Step "2/10" "Поиск установленного Python (подходят 3.11, 3.12, 3.13)"
# предпочтение: 3.12 (лучше всего с готовыми сборками библиотек), затем 3.11, затем 3.13
function Get-PyInfo($cmd, $pre) {
    try {
        $out = & $cmd @pre -c "import sys; print('%d.%d|%s|%s' % (sys.version_info[0], sys.version_info[1], sys.executable, sys.base_prefix))" 2>$null
        if ($LASTEXITCODE -eq 0 -and $out) { $p = ([string]$out).Trim().Split("|"); return @{ ver = $p[0]; exe = $p[1]; prefix = $p[2] } }
    } catch { }
    return $null
}
$found = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    foreach ($v in @("3.12", "3.11", "3.13")) { $i = Get-PyInfo "py" @("-$v"); if ($i) { $found += $i } }
}
# без лаунчера py: python.exe из PATH (кроме заглушки Microsoft Store в WindowsApps)
foreach ($c in (Get-Command python.exe -All -ErrorAction SilentlyContinue)) {
    if ($c.Source -notmatch "WindowsApps") { $i = Get-PyInfo $c.Source @(); if ($i) { $found += $i } }
}
# обычные места установки
foreach ($d in (Get-ChildItem "$env:ProgramFiles\Python3*", "$env:LOCALAPPDATA\Programs\Python\Python3*" -Directory -ErrorAction SilentlyContinue)) {
    $exe = Join-Path $d.FullName "python.exe"
    if (Test-Path $exe) { $i = Get-PyInfo $exe @(); if ($i) { $found += $i } }
}
$rank = @{ "3.12" = 0; "3.11" = 1; "3.13" = 2 }
$usable = $found | Where-Object { $rank.ContainsKey($_.ver) } | Sort-Object { $rank[$_.ver] }, { [int](HasNonAscii $_.exe) }
if ($found) {
    Write-Host "  найдено:"
    $found | ForEach-Object { "    $($_.ver)  $($_.exe)" } | Sort-Object -Unique | ForEach-Object { Write-Host $_ }
}
if (-not $usable) {
    $vers = ($found | ForEach-Object { $_.ver } | Sort-Object -Unique) -join ", "
    if ($vers) { Write-Host "  есть Python $vers, но нужна версия 3.11–3.13 (для версий ниже 3.11 и для 3.14 нет всех нужных библиотек)" -ForegroundColor Yellow }
    Fail ("Установите Python 3.12: winget install Python.Python.3.12 --scope machine`n" +
          "затем перезапустите PowerShell и запустите снова:`n" + (Invoke-RerunHint))
}
$basePyExe = ($usable | Select-Object -First 1).exe
$basePy = ($usable | Select-Object -First 1).prefix
Write-Host "  использую: Python $(($usable | Select-Object -First 1).ver)  $basePyExe" -ForegroundColor Green
Write-Host "  (сам Python не меняется: библиотеки ставятся в отдельное окружение внутри папки программы)" -ForegroundColor DarkGray
if (HasNonAscii $basePy) {
    Write-Host ("  Внимание: Python установлен в папку с кириллицей. Если что-то не заработает, поставьте его для всех пользователей:`n" +
                "    winget install Python.Python.3.12 --scope machine") -ForegroundColor Yellow
}
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
    Write-Host "  ffmpeg не найден, ставлю через winget..." -ForegroundColor Yellow
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail ("winget недоступен, поставьте ffmpeg вручную (https://www.gyan.dev/ffmpeg/builds/)`n" +
              "и добавьте папку с ffmpeg.exe в PATH, затем запустите снова:`n" + (Invoke-RerunHint))
    }
    winget install --id Gyan.FFmpeg -e --accept-source-agreements --accept-package-agreements
    if ($LASTEXITCODE -ne 0) {
        Fail ("winget не смог установить ffmpeg (код $LASTEXITCODE).`n" +
              "Проверьте вручную: winget install --id Gyan.FFmpeg -e`n" +
              "Затем запустите снова:`n" + (Invoke-RerunHint))
    }
    # winget дописывает PATH в реестр; подхватываем его в текущем окне, чтобы не требовать перезапуска
    Update-SessionPath
    if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) {
        Fail ("ffmpeg установлен, но не виден в этом окне PowerShell.`n" +
              "Перезапустите PowerShell и запустите снова:`n" + (Invoke-RerunHint))
    }
    Write-Host "  ffmpeg готов" -ForegroundColor Green
}
}

Step "3/10" "Папка программы и права доступа"
$isNew = -not (Test-Path $Target)
New-Item -ItemType Directory -Path $Target -Force | Out-Null
if (-not $insideProfile) {
    # вне профиля Windows по умолчанию дала бы право изменения всем пользователям — закрываем
    $me = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    icacls $Target /inheritance:r /grant:r "*${me}:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" /Q | Out-Null
    Check "права на папку"
    Write-Host "  доступ: только вы, SYSTEM и администраторы"
} else {
    Write-Host "  папка в вашем профиле: доступ только у вас (права профиля)"
}

# папка личных данных: создаётся, прав доступа как у папки программы; установщик её не перезаписывает и не чистит
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null
if (-not $DataDir.StartsWith($env:USERPROFILE + "\", [StringComparison]::OrdinalIgnoreCase)) {
    $me = [Security.Principal.WindowsIdentity]::GetCurrent().User.Value
    icacls $DataDir /inheritance:r /grant:r "*${me}:(OI)(CI)F" "*S-1-5-18:(OI)(CI)F" "*S-1-5-32-544:(OI)(CI)F" /Q | Out-Null
    Check "права на папку данных"
    Write-Host "  данные: $DataDir (доступ: только вы, SYSTEM и администраторы)"
} else {
    Write-Host "  данные: $DataDir (в вашем профиле)"
}

Step "4/10" "Копирование файлов программы"
if ($Source -ne $Target) {
    foreach ($f in @("server.py", "worker.py", "bench.py", "runinfo.py", "backends.py", "paths.py", "recordings.py", "storage.py", "documents.py", "workflow.py", "product_policy.py", "model_tasks.py", "llm_provider.py", "download_hint.py", "transcript.py",
                     "diar_fast.py", "vocab.py", "voices.py", "gender.py", "words.py", "hotspots.py", "quotes.py", "conspect.py",
                     "review_pack.py", "freeform.py", "jobslots.py", "settings.py", "migrate.py", "data_io.py", "core.py", "confirm.py", "tmpfiles.py", "selfcheck.py", "rt.py", "voice_io.py", "voice_review.py", "review_page.py", "workflow_page.py", "cards.html", "workflow.html", "README.md", "install.ps1")) {
        Copy-Item (Join-Path $Source $f) (Join-Path $Target $f) -Force
    }
    $docsTarget = Join-Path $Target "docs"
    New-Item -ItemType Directory -Path $docsTarget -Force | Out-Null
    Copy-Item (Join-Path $Source "docs\model-download-certificate.md") (Join-Path $docsTarget "model-download-certificate.md") -Force
    Copy-Item (Join-Path $Source "docs\2026-11-17-p3-workflow-handoff.md") (Join-Path $docsTarget "2026-11-17-p3-workflow-handoff.md") -Force
    Copy-Item (Join-Path $Source "docs\2026-11-17-p4-opencode-openai-handoff.md") (Join-Path $docsTarget "2026-11-17-p4-opencode-openai-handoff.md") -Force
    Copy-Item (Join-Path $Source "docs\2026-11-17-p5-screen-handoff.md") (Join-Path $docsTarget "2026-11-17-p5-screen-handoff.md") -Force
    # tools — проверка качества и check_repo.py; tests — для самопроверки (self_check);
    # examples — вымышленный пример группы. Это копии комплекта программы, а личные данные
    # лежат в папке данных, поэтому при обновлении папки заменяются целиком: иначе старые
    # тесты и инструменты остаются рядом с новыми и self_check проверяет устаревший набор.
    foreach ($sub in @("tools", "tests", "examples")) {
        $sd = Join-Path $Source $sub
        if (-not (Test-Path $sd)) { continue }
        $dd = Join-Path $Target $sub
        $srcFiles = @(Get-ChildItem $sd -Recurse -File | ForEach-Object { $_.FullName.Substring($sd.Length + 1) })
        # файлы, которых нет в комплекте, могли добавить вы сами: без согласия их не удаляем
        $extra = @()
        if (Test-Path $dd) {
            $extra = @(Get-ChildItem $dd -Recurse -File -ErrorAction SilentlyContinue |
                ForEach-Object { $_.FullName.Substring($dd.Length + 1) } |
                Where-Object { $srcFiles -notcontains $_ -and $_ -notlike "$sub\*" -and
                               ($_ -split '\\') -notcontains "__pycache__" -and $_ -notlike "*.pyc" })
        }
        $replace = $true
        if ($extra.Count) {
            $shown = ($extra | Select-Object -First 5) -join "; "
            if ($extra.Count -gt 5) { $shown += "; и ещё " + ($extra.Count - 5) }
            Write-Host "  в папке $sub есть файлы, которых нет в комплекте программы ($($extra.Count)):" -ForegroundColor Yellow
            Write-Host "    $shown" -ForegroundColor Yellow
            Write-Host "  при обновлении папка $sub заменяется целиком, чтобы self_check проверял актуальный набор" -ForegroundColor Yellow
            $ans = Read-Host "  Удалить эти файлы? Enter или y — удалить, n — оставить (Y/n)"
            $replace = -not ($ans -match '^(n|н)')
            if (-not $replace) {
                Write-Host "  файлы оставлены: файлы программы будут перезаписаны, устаревшие останутся, self_check может показывать старые проверки" -ForegroundColor Yellow
            }
        }
        if ($replace) {
            if (Test-Path $dd) { Remove-Item $dd -Recurse -Force }
            Copy-Item $sd $dd -Recurse -Force
        } else {
            New-Item -ItemType Directory -Path $dd -Force | Out-Null
            Copy-Item (Join-Path $sd "*") $dd -Recurse -Force
        }
        # вложенные дубли прошлых версий установщика и кэш байткода — не данные пользователя
        $dup = Join-Path $dd $sub
        if (Test-Path $dup) { Remove-Item $dup -Recurse -Force -ErrorAction SilentlyContinue }
        Get-ChildItem $dd -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
            Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    }
    # данные (профили, словари, настройки, задачи) лежат в папке данных, установщик их не копирует и не трогает;
    # старые profiles\, config.json, jobs\ из папки программы переносит явная команда migrate_data (или migrate.py)
} else { Write-Host "  запуск из папки программы, копировать нечего" }
Set-Location $Target

# все кэши — внутри папки программы
$env:PIP_CACHE_DIR = Join-Path $Target "cache\pip"
$env:HF_HOME       = Join-Path $Target "cache\huggingface"
$env:TORCH_HOME    = Join-Path $Target "cache\torch"

if ($AcceptanceCopyOnly) {
    Write-Host "  режим приёмки: venv, зависимости и HF-проверки пропущены"
} else {
Step "5/10" "Виртуальное окружение"
$py = Join-Path $Target ".venv\Scripts\python.exe"
# версии установленных пакетов (по имени дистрибутива; пустая строка — не установлен)
function Get-PkgVersions([string[]]$names) {
    $code = "import importlib.metadata as m,sys;d={(x.metadata['Name'] or '').lower().replace('_','-'):x.version for x in m.distributions()};[print(n+'='+d.get(n,'')) for n in sys.argv[1:]]"
    $map = @{}
    foreach ($line in @(& $py -c $code @names 2>$null)) {
        $kv = ([string]$line).Split("=", 2)
        if ($kv.Count -eq 2) { $map[$kv[0]] = $kv[1].Trim() }
    }
    return $map
}
function Get-PipFreeze { return @(& $py -m pip freeze --all 2>$null | ForEach-Object { [string]$_ }) }
$freezeBefore = $null
if (Test-Path $py) {
    $venvVer = (& $py -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
    Write-Host "  окружение уже есть (Python $venvVer), обновляю библиотеки"
    # снимок пакетов для отката: cache\pip-freeze, хранятся 10 последних
    $freezeBefore = Get-PipFreeze
    if ($freezeBefore.Count) {
        $freezeDir = Join-Path $Target "cache\pip-freeze"
        New-Item -ItemType Directory -Path $freezeDir -Force | Out-Null
        $freezeFile = Join-Path $freezeDir ("before-update-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".txt")
        WriteUtf8NoBom $freezeFile (($freezeBefore -join "`n") + "`n")
        Write-Host "  список пакетов до обновления: $freezeFile"
        Get-ChildItem -LiteralPath $freezeDir -Filter "before-update-*.txt" -File | Sort-Object Name -Descending |
            Select-Object -Skip 10 | Remove-Item -Force -ErrorAction SilentlyContinue
    }
} else {
    & $basePyExe -m venv (Join-Path $Target ".venv"); Check "venv"
}
& $py -m pip install --upgrade pip wheel; Check-Pip "pip"

Step "6/10" "PyTorch (процессор), pyannote, GigaAM, MCP"
& $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu; Check-Pip "PyTorch"
$step6 = @("pyannote.audio>=4.0", "pytorch-metric-learning==2.9.0", "soundfile", "truststore", "mcp[cli]")
# GigaAM ставится из архива ветки main и тянет обычный onnxruntime, который ниже всё равно заменяется:
# если GigaAM уже стоит, не переустанавливаем (чтобы обновить GigaAM, удалите пакет и запустите установку)
$gigaamVer = (Get-PkgVersions @("gigaam"))["gigaam"]
if ($gigaamVer) { Write-Host "  GigaAM $gigaamVer уже установлен — не переустанавливаю" }
else { $step6 += "gigaam @ https://github.com/salute-developers/GigaAM/archive/refs/heads/main.zip" }
& $py -m pip install @step6; Check-Pip "pyannote/GigaAM"
# pytorch-metric-learning 3.x требует numpy>=2.5.3, а OpenVINO 2025.4.1 — numpy<2.4.0: держим совместимую 2.9.0

Step "7/10" "OpenVINO для видеокарты Arc"
& $py -m pip install "onnx-asr[hub]" "openvino==2025.4.1"; Check-Pip "onnx-asr/OpenVINO"
# onnxruntime, onnxruntime-directml и onnxruntime-openvino дают один и тот же модуль:
# убираем обычные сборки и ставим сборку с OpenVINO последней — только если это нужно
$onnx = Get-OnnxPlan (Get-PkgVersions @("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu", "onnxruntime-openvino")) "1.24.1"
if ($onnx.remove.Count) { & $py -m pip uninstall -y @($onnx.remove) }
if ($onnx.reinstall) {
    & $py -m pip install --force-reinstall --no-deps "onnxruntime-openvino==1.24.1"; Check-Pip "onnxruntime-openvino"
} else { Write-Host "  onnxruntime-openvino 1.24.1 уже установлен, других сборок onnxruntime нет — не переустанавливаю" }
# под Windows модуль OpenVINO в onnxruntime берёт библиотеки из пакета openvino — версии должны совпадать
# (onnxruntime-openvino 1.24.x собран с OpenVINO 2025.4.1; с 2026.x — «ошибка 127» при загрузке)
# проверяем именно версию: одной проверки «import openvino» мало, потому что любая версия импортируется
$ov = & $py -c "import openvino; print(openvino.__version__.split('-')[0])" 2>$null
if ($ov -ne "2025.4.1") {
    Write-Host "  OpenVINO $ov не подходит — ставлю 2025.4.1"
    & $py -m pip install "openvino==2025.4.1"; Check-Pip "OpenVINO 2025.4.1"
} else { Write-Host "  OpenVINO 2025.4.1 — версия подходит" }
# faster-whisper (CTranslate2) — альтернативный движок Whisper, выбирается тестом скорости (bench).
# Ставим с --no-deps: он тянет обычный onnxruntime, взаимно исключаемый с onnxruntime-openvino;
# onnxruntime нужен faster-whisper только для встроенного VAD, который не используется.
# huggingface_hub уже поставлен с pyannote.audio.
& $py -m pip install --no-deps "faster-whisper==1.2.1"; Check-Pip "faster-whisper"
& $py -m pip install "ctranslate2==4.9.0" "tokenizers" "av"; Check-Pip "CT2/tokenizers/av"
# после смены версии OpenVINO сохранённые модели Whisper пересоберутся при первом запуске (это нормально)
# словарь русского языка: свободные ответы в карточках — отличать опечатку от настоящего слова
& $py -m pip install pymorphy3 pymorphy3-dicts-ru; Check-Pip "pymorphy3"
if ($freezeBefore) {
    $changes = Compare-PipFreeze $freezeBefore (Get-PipFreeze)
    if (-not $changes.Count) { Write-Host "  пакеты окружения не изменились" }
    else {
        Write-Host "  изменились пакеты окружения ($($changes.Count)):" -ForegroundColor Yellow
        $changes | Select-Object -First 20 | ForEach-Object { Write-Host "    $_" }
        if ($changes.Count -gt 20) { Write-Host "    … и ещё $($changes.Count - 20)" }
        Write-Host "  откат к прежним версиям: & '$py' -m pip install -r '$freezeFile' (вместе с прежней версией программы, см. docs/development-and-installation.md)" -ForegroundColor DarkGray
    }
}
& $py -c "import paths, backends as b; print('  OpenVINO видит:', b.openvino_devices()); print('  ONNX Runtime:', b.available_providers())"
Write-Host "  нужны CPU и OpenVINOExecutionProvider; GPU желателен на Intel Arc; NPU не обязателен (на Meteor Lake он для GigaAM не используется)" -ForegroundColor DarkGray
# OpenVINO GPU/NPU — плагины Intel: на графике AMD/NVIDIA их не будет, и это не ошибка.
$nonIntel = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -and $_.Name -notmatch 'Intel' }
if ($nonIntel) {
    $names = ($nonIntel | ForEach-Object { $_.Name }) -join ', '
    Write-Host "  Видео: $names." -ForegroundColor DarkGray
    Write-Host "  OpenVINO GPU/NPU работают только на Intel — отсутствие GPU в списке выше для этой видеокарты не ошибка и не требует драйверов:" -ForegroundColor DarkGray
    Write-Host "  тест скорости сам выберет процессор. Для Whisper обычно быстрее whisper-ct2 (faster-whisper): bench.py --only whisper-ct2,whisper-cpu." -ForegroundColor DarkGray
    Write-Host "  Задействовать видеокарту можно через DirectML (пакет onnxruntime-directml) — замена пакета по согласованию: bench.py --only dml-gpu,whisper-dml" -ForegroundColor DarkGray
}

Step "8/10" "Токен Hugging Face"
# при обновлении приложение ИИ-агента уже хранит рабочий токен — второй раз вводить не нужно
$savedToken = ""
foreach ($cp in (Find-ClaudeConfig)) {
    if (Test-Path $cp) {
        try {
            $saved = (Get-Content $cp -Raw -Encoding UTF8 | ConvertFrom-Json).mcpServers.$Name
            if ($saved -and $saved.env -and $saved.env.HF_TOKEN -and
                $saved.env.HF_TOKEN -ne "ВСТАВЬТЕ_СЮДА_hf_ТОКЕН") { $savedToken = [string]$saved.env.HF_TOKEN }
        } catch { }
    }
}
$tokenMode = Get-TokenMode $Token ([bool]$KeepSavedToken) $savedToken $nonInteractive
if ($tokenMode -eq "ask") {
    $hint = if ($savedToken) { "Enter — оставить сохранённый токен" } else { "Enter — пропустить" }
    $sec = Read-Host "Вставьте токен Hugging Face (hf_..., тип Read). $hint" -AsSecureString
    if ($sec) { $Token = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec)) }
}
$Token = ([string]$Token).Trim()
if ($Token) {
    $h = @{ Authorization = "Bearer $Token" }
    try {
        $who = Invoke-RestMethod "https://huggingface.co/api/whoami-v2" -Headers $h -TimeoutSec 20
        Write-Host "  токен рабочий, пользователь: $($who.name)"
        try {
            Invoke-WebRequest "https://huggingface.co/pyannote/speaker-diarization-community-1/resolve/main/config.yaml" `
                -Headers $h -Method Head -UseBasicParsing -TimeoutSec 20 | Out-Null
            Write-Host "  доступ к pyannote community-1 есть"
        } catch {
            Write-Host ("  нет доступа к модели pyannote: откройте https://huggingface.co/pyannote/speaker-diarization-community-1`n" +
                        "  и примите условия (кнопка вверху страницы), потом всё заработает без переустановки") -ForegroundColor Yellow
        }
    } catch { Write-Host "  токен не принят Hugging Face — проверьте, что скопировали его целиком" -ForegroundColor Yellow }
} elseif ($savedToken) {
    Write-Host "  пропущено: останется сохранённый токен из настроек приложения ИИ-агента" -ForegroundColor DarkGray
} else {
    Write-Host "  пропущено: допишите HF_TOKEN в настройки приложения ИИ-агента позже (см. README)" -ForegroundColor Yellow
}
}

Step "9/10" "Подключение к приложению ИИ-агента (например, Claude Desktop)"
if ($nonInteractive -and -not $SkipClaudeConfig -and (Get-Process -Name "Claude" -ErrorAction SilentlyContinue)) {
    # без терминала некому закрыть приложение и нажать Enter: раньше здесь был бесконечный цикл
    Write-Host ("  пропущено: приложение ИИ-агента запущено, а терминала нет. Если запись нужна — закройте приложение " +
                "и запустите установщик ещё раз в терминале (или с -SkipClaudeConfig, если запись уже есть)") -ForegroundColor Yellow
    $SkipClaudeConfig = $true
} elseif ($SkipClaudeConfig) {
    Write-Host "  пропущено (-SkipClaudeConfig)"
}
if (-not $SkipClaudeConfig) {
    # Приложение при выходе сохраняет настройки поверх файла и затирает нашу запись,
    # поэтому писать можно только при полностью закрытом приложении.
    while (Get-Process -Name "Claude" -ErrorAction SilentlyContinue) {
        Write-Host "  Приложение ИИ-агента (например, Claude Desktop) запущено. Закройте его полностью: правой кнопкой по значку в трее → Quit (Выход)." -ForegroundColor Yellow
        $null = Read-Host "  Когда закроете, нажмите Enter"
        Start-Sleep -Seconds 2
    }
    $server = Join-Path $Target "server.py"
    $entry = [pscustomobject]@{ command = [string]$py; args = @([string]$server); env = [pscustomobject]@{ HF_TOKEN = $(if ($Token) { $Token } else { "ВСТАВЬТЕ_СЮДА_hf_ТОКЕН" }); RUS_TRANSCRIBER_DATA = [string]$DataDir } }
    foreach ($cp in (Find-ClaudeConfig)) {
        Edit-ClaudeConfig $cp {
            param($s)
            $old = $s.$Name
            if ($old -and $old.env -and $old.env.HF_TOKEN -and -not $Token) { $entry.env.HF_TOKEN = $old.env.HF_TOKEN }
            $s | Add-Member -NotePropertyName $Name -NotePropertyValue $entry -Force
        }
        $chk = $null
        try { $chk = (Get-Content $cp -Raw -Encoding UTF8 | ConvertFrom-Json).mcpServers.$Name } catch { }
        if ($chk -and ([string]$chk.args[0]) -eq $server) { Write-Host "  записано и проверено: $cp" -ForegroundColor Green }
        else { Fail "Запись в $cp не сохранилась. Пришлите этот файл (без токена) для разбора." }
    }
    Write-Host "  Теперь запустите приложение ИИ-агента (например, Claude Desktop)." -ForegroundColor Yellow
}

if ($AcceptanceCopyOnly) {
    Write-Host "Копирование runtime-файлов для приёмки завершено; зависимости и настройки не менялись."
    exit 0
}

# старые кэши от первой версии в профиле можно удалить
$old = @("$env:USERPROFILE\.cache\gigaam", "$env:USERPROFILE\.cache\huggingface") | Where-Object { Test-Path $_ }
if ($old) {
    Write-Host "`nНайдены кэши прошлой установки (их можно удалить, если они не нужны другим программам):" -ForegroundColor DarkGray
    $old | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
}

Step "10/10" "Самопроверка (тесты в отдельной временной папке, проверка файлов, морфология)"
$selfCheckOk = $true
if ($SkipSelfCheck) {
    Write-Host "  пропущено (-SkipSelfCheck); запустить позже: & '$py' '$Target\selfcheck.py'"
} else {
    # selfcheck.py сам изолирует папку данных и TEMP для тестов; настоящие данные не трогаются
    $prevEnc = [Console]::OutputEncoding
    $prevUtf8 = $env:PYTHONUTF8; $prevIo = $env:PYTHONIOENCODING
    try {
        [Console]::OutputEncoding = New-Object Text.UTF8Encoding $false
        $env:PYTHONUTF8 = "1"; $env:PYTHONIOENCODING = "utf-8"
        $raw = (& $py -B (Join-Path $Target "selfcheck.py") 2>$null) -join "`n"
    } finally {
        [Console]::OutputEncoding = $prevEnc; $env:PYTHONUTF8 = $prevUtf8; $env:PYTHONIOENCODING = $prevIo
    }
    $sc = $null
    try { $sc = $raw | ConvertFrom-Json } catch { }
    if (-not $sc) {
        $selfCheckOk = $false
        Write-Host "  самопроверка не дала результата; запустите вручную: & '$py' '$Target\selfcheck.py'" -ForegroundColor Yellow
    } else {
        $selfCheckOk = [bool]$sc.ok
        $t = $sc.tests
        $line = "  {0}: тестов {1}, падений {2}, ошибок {3}, пропусков {4}" -f $sc.summary, $t.ran, $t.failures, $t.errors, $t.skipped
        if ($selfCheckOk) { Write-Host $line -ForegroundColor Green } else { Write-Host $line -ForegroundColor Yellow }
        foreach ($f in @($t.failed)) { if ($f) { Write-Host "    $($f.kind): $($f.test)" -ForegroundColor Yellow } }
        if ($sc.check_repo -and $sc.check_repo.partial) { Write-Host "  проверка файлов частичная: $($sc.check_repo.note)" -ForegroundColor DarkGray }
    }
}

Write-Host "`nГотово. Программа, модели и кэши: $Target" -ForegroundColor Green
Write-Host "Личные данные: $DataDir (установщик их не трогает; перенос между компьютерами — export_group / import_group)" -ForegroundColor Green
Write-Host "Дальше — тест скорости. Попросите ИИ-агента (например, Claude Desktop): «запусти тест скорости на C:\...\запись.mp3»,"
Write-Host "или вручную:  `$env:HF_TOKEN='hf_...'; & '$py' '$Target\bench.py' 'C:\путь\к\записи.mp3'"
if ($isUpdate) { Write-Host "После обновления перезапустите приложение ИИ-агента (например, Claude Desktop), чтобы сервер подхватил новые файлы." -ForegroundColor Yellow }
if (-not $selfCheckOk) {
    Write-Host "Установка завершена, но самопроверка нашла проблемы (см. шаг 10). Код выхода 2." -ForegroundColor Yellow
    exit 2
}
