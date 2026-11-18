# Синтетическая проверка удаления и защит install.ps1 (запускает tests/test_install_uninstall.py).
# Всё создаётся во временной папке; реальные настройки и данные не затрагиваются.
param([string]$Script)
$ErrorActionPreference = "Stop"
$sep = [IO.Path]::DirectorySeparatorChar
$results = @()
function Check($name, $cond) { $script:results += [pscustomobject]@{ test = $name; ok = [bool]$cond } }
function global:Get-CimInstance { param([string]$ClassName, [string]$Filter, $ErrorAction)
    if ($Filter) { return $null }
    return $global:FakeProcs }

function New-Sandbox($tag) {
    $root = Join-Path ([IO.Path]::GetTempPath()) ("uninst-" + $tag + "-" + [guid]::NewGuid().ToString("N").Substring(0,6))
    $env:LOCALAPPDATA = Join-Path $root "local"; $env:APPDATA = Join-Path $root "roaming"
    $env:TEMP = Join-Path $root "temp"; $env:USERPROFILE = $root
    foreach ($d in @($env:LOCALAPPDATA, $env:APPDATA, $env:TEMP)) { New-Item -ItemType Directory $d -Force | Out-Null }
    $t = Join-Path $root "program"; $dd = Join-Path $root "data"; $out = Join-Path $root "outside"
    foreach ($d in @("$t/.venv/lib", "$t/cache/hf", "$t/tests", "$t/meeting.rus-transcriber", "$dd/groups/g", "$dd/jobs/r.review",
                     "$dd/exports", "$dd/_backup-import-1/groups", "$dd/bench", $out)) { New-Item -ItemType Directory $d -Force | Out-Null }
    Set-Content "$t/server.py" "x"; Set-Content "$t/core.py" "x"; Set-Content "$t/tests/t.py" "x"
    Set-Content "$t/.venv/lib/sample.wav" "pkg"; Set-Content "$t/cache/hf/model.bin" "m"
    Set-Content "$t/my-recording.mp3" "audio"; Set-Content "$t/my-recording.transcript.json" "{}"
    Set-Content "$t/meeting.rus-transcriber/meeting.md" "doc"
    Set-Content "$out/precious.txt" "keep"
    try { New-Item -ItemType SymbolicLink -Path "$t/link-out" -Target $out -ErrorAction Stop | Out-Null; $global:HasLink = $true }
    catch { $global:HasLink = $false }      # нет прав на ссылки — эту часть проверки пропускаем
    Set-Content "$dd/config.json" "{}"; Set-Content "$dd/groups/g/voices.json" "{}"; Set-Content "$dd/groups/g/vocab.json" "{}"
    Set-Content "$dd/jobs/1.json" "{}"; Set-Content "$dd/jobs/1.log" "log"; Set-Content "$dd/jobs/r.review/clip.mp3" "a"
    Set-Content "$dd/state.sqlite3" "db"; Set-Content "$dd/exports/e.zip" "z"; Set-Content "$dd/_backup-import-1/groups/v.json" "b"
    Set-Content "$dd/bench/bench_report.md" "r"
    $cfgDir = Join-Path $env:APPDATA "Claude"; New-Item -ItemType Directory $cfgDir -Force | Out-Null
    $cfg = Join-Path $cfgDir "claude_desktop_config.json"
    $j = @{ mcpServers = @{ "rus-transcriber" = @{ command = "py"; args = @("$t/server.py"); env = @{ HF_TOKEN = "hf_dummy"; RUS_TRANSCRIBER_DATA = $dd } };
                            "other" = @{ command = "x" } } } | ConvertTo-Json -Depth 8
    Set-Content $cfg $j; Set-Content "$cfg.bak-20260101-000000" $j
    return @{ root = $root; t = $t; dd = $dd; out = $out; cfg = $cfg }
}
function Run($sb, [string[]]$extra, [string]$scriptPath = $Script) {
    $a = @{ Uninstall = $true; Target = $sb.t; DataDir = $sb.dd }
    foreach ($x in $extra) { $a[$x.TrimStart("-")] = $true }
    $o = "" | & $scriptPath @a *>&1 | Out-String
    return @{ code = $LASTEXITCODE; out = $o }
}
$global:FakeProcs = @()

# A: без ключей и без терминала — ничего не меняется
$s = New-Sandbox "a"; $r = Run $s @()
Check "A refuses without flags (code 1)" ($r.code -eq 1)
Check "A program intact" (Test-Path "$($s.t)/server.py")
Check "A config entry intact" ((Get-Content $s.cfg -Raw) -match "rus-transcriber")

# B: программа работает — отказ, ничего не меняется
$s = New-Sandbox "b"
$global:FakeProcs = @([pscustomobject]@{ ProcessId = 99999; Name = "python.exe"; ExecutablePath = "/usr/bin/python3"; CommandLine = "python $($s.t)/server.py" })
$r = Run $s @("-Yes", "-RemoveData")
$global:FakeProcs = @()
Check "B refuses while running (code 1)" ($r.code -eq 1)
Check "B program intact" (Test-Path "$($s.t)/server.py")
Check "B data intact" (Test-Path "$($s.dd)/groups/g/voices.json")

# C: полное удаление
$s = New-Sandbox "c"; $r = Run $s @("-Yes", "-RemoveData", "-RemoveConfigBackups")
Check "C exit 0" ($r.code -eq 0)
Check "C program code removed" (-not (Test-Path "$($s.t)/server.py") -and -not (Test-Path "$($s.t)/tests"))
Check "C venv and cache removed (incl. package wav)" (-not (Test-Path "$($s.t)/.venv") -and -not (Test-Path "$($s.t)/cache"))
Check "C user audio in program kept" (Test-Path "$($s.t)/my-recording.mp3")
Check "C transcript in program kept" (Test-Path "$($s.t)/my-recording.transcript.json")
Check "C sidecar folder kept" (Test-Path "$($s.t)/meeting.rus-transcriber/meeting.md")
Check "C link removed, target untouched" ((-not $global:HasLink) -or (-not (Test-Path "$($s.t)/link-out") -and (Test-Path "$($s.out)/precious.txt")))
Check "C service data removed" (-not (Test-Path "$($s.dd)/groups") -and -not (Test-Path "$($s.dd)/config.json") -and -not (Test-Path "$($s.dd)/bench") -and -not (Test-Path "$($s.dd)/jobs/1.log"))
Check "C db, exports, backups kept" ((Test-Path "$($s.dd)/state.sqlite3") -and (Test-Path "$($s.dd)/exports/e.zip") -and (Test-Path "$($s.dd)/_backup-import-1/groups/v.json"))
Check "C review clip in jobs kept" (Test-Path "$($s.dd)/jobs/r.review/clip.mp3")
$c = Get-Content $s.cfg -Raw | ConvertFrom-Json
Check "C entry removed, other server kept" (-not ($c.mcpServers.PSObject.Properties.Name -contains "rus-transcriber") -and ($c.mcpServers.PSObject.Properties.Name -contains "other"))
Check "C config backups removed" (-not (Get-ChildItem "$($s.cfg).bak-*" -ErrorAction SilentlyContinue))
Check "C report lists kept files" ($r.out -match "сохранены ваши файлы")

# D: программа удаляется, данные и копии — нет
$s = New-Sandbox "d"; $r = Run $s @("-Yes", "-KeepData")
Check "D exit 0" ($r.code -eq 0)
Check "D program removed" (-not (Test-Path "$($s.t)/server.py"))
Check "D data untouched" ((Test-Path "$($s.dd)/groups/g/voices.json") -and (Test-Path "$($s.dd)/config.json"))
Check "D backups kept (incl. new one)" (@(Get-ChildItem "$($s.cfg).bak-*").Count -ge 2)

# E: запуск из папки программы — работа из временной копии
$s = New-Sandbox "e"; Copy-Item $Script "$($s.t)/install.ps1"
$r = Run $s @("-Yes", "-KeepData") "$($s.t)/install.ps1"
Check "E exit 0" ($r.code -eq 0)
Check "E program removed" (-not (Test-Path "$($s.t)/server.py") -and -not (Test-Path "$($s.t)/install.ps1"))
Check "E temp copy cleaned" (-not (Get-ChildItem $env:TEMP -Filter "rus-transcriber-uninstall-*" -ErrorAction SilentlyContinue))

# F: в папке нет программы — ничего не удаляется из неё
$s = New-Sandbox "f"; Remove-Item "$($s.t)/server.py"; $r = Run $s @("-Yes", "-KeepData")
Check "F folder without server.py untouched" (Test-Path "$($s.t)/core.py")
Check "F config entry kept" ((Get-Content $s.cfg -Raw) -match "rus-transcriber")

# G: битый JSON настроек — отказ до изменений
$s = New-Sandbox "g"; Set-Content $s.cfg "{ broken"; $r = Run $s @("-Yes", "-RemoveData")
Check "G refuses on broken config" ($r.code -eq 1 -and (Test-Path "$($s.t)/server.py") -and (Test-Path "$($s.dd)/config.json"))

# H: в папке программы есть .git — удаление отказывает, ничего не трогает
$s = New-Sandbox "h"; New-Item -ItemType Directory "$($s.t)/.git" | Out-Null; Set-Content "$($s.t)/.git/HEAD" "ref"
$r = Run $s @("-Yes", "-RemoveData", "-RemoveConfigBackups")
Check "H uninstall refuses with .git (code 1)" ($r.code -eq 1 -and $r.out -match "рабочая копия")
Check "H program, git, data, entry intact" ((Test-Path "$($s.t)/server.py") -and (Test-Path "$($s.t)/.git/HEAD") -and (Test-Path "$($s.dd)/config.json") -and ((Get-Content $s.cfg -Raw) -match "rus-transcriber"))

# I: установка из другой папки в папку с .git — отказ до любых шагов
$s = New-Sandbox "i"; New-Item -ItemType Directory "$($s.t)/.git" | Out-Null
$src = Join-Path $s.root "src"; New-Item -ItemType Directory $src | Out-Null; Copy-Item $Script "$src/install.ps1"
$o = "" | & "$src/install.ps1" -Target $s.t -DataDir $s.dd -SkipClaudeConfig *>&1 | Out-String
$code = $LASTEXITCODE
Check "I install exit code 1" ($code -eq 1)
Check "I install message about git working copy" ($o -match "рабочая копия разработки")
Check "I install stopped before step 1" ($o -notmatch "Проверка места")
if (-not ($code -eq 1 -and $o -match "рабочая копия разработки" -and $o -notmatch "Проверка места")) {
    $global:Diag = "---- I: code=$code, output ----`n" + ($o.Substring([Math]::Max(0, $o.Length - 3000)))
}
Check "I target untouched" ((Test-Path "$($s.t)/core.py") -and -not (Test-Path "$($s.t)/install.ps1"))

$results | Format-Table -AutoSize | Out-String -Width 200
if ($global:Diag) { $global:Diag }
$bad = @($results | Where-Object { -not $_.ok }).Count
"FAILED: $bad"
if ($bad) { exit 1 }
exit 0
