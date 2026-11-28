# Синтетическая проверка обновления install.ps1 (запускает tests/test_install_update.py).
# Помощники берутся из install.ps1 разбором кода; сценарии запуска останавливаются до шага 1
# или на нём (латинский путь не задан намеренно), поэтому pip, сеть и настоящие данные не затрагиваются.
param([string]$Script)
# как в install.ps1: сообщения git в stderr не должны прерывать проверку
$ErrorActionPreference = "Continue"
$results = @()
function Check($name, $cond) { $script:results += [pscustomobject]@{ test = $name; ok = [bool]$cond } }
function global:Get-CimInstance { param([string]$ClassName, [string]$Filter, $ErrorAction)
    if ($Filter) { return $null }
    return $global:FakeProcs }
$global:FakeProcs = @()

# временная папка на весь прогон; переменные окружения указывают внутрь неё
$base = Join-Path ([IO.Path]::GetTempPath()) ("upd-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory $base | Out-Null
$env:LOCALAPPDATA = Join-Path $base "local"; $env:APPDATA = Join-Path $base "roaming"; $env:USERPROFILE = $base
$env:TEMP = Join-Path $base "temp"; $env:TMP = $env:TEMP
foreach ($d in @($env:LOCALAPPDATA, $env:APPDATA, $env:TEMP)) { New-Item -ItemType Directory $d -Force | Out-Null }

# ---------- помощники из install.ps1 ----------
$tokens = $null; $errs = $null
$ast = [Management.Automation.Language.Parser]::ParseFile($Script, [ref]$tokens, [ref]$errs)
Check "installer parses" (-not $errs)
$wanted = @("Get-ProgramProcesses", "Split-ProgramProcesses", "Get-ServerVersion", "Get-SourceState",
            "Get-OnnxPlan", "Get-TokenMode", "Compare-PipFreeze", "Get-ExtraFilesMode", "Get-ExtraFilesAnswer")
$defs = @($ast.FindAll({ param($n) $n -is [Management.Automation.Language.FunctionDefinitionAst] }, $false) |
    Where-Object { $wanted -contains $_.Name })
Check "helpers found" ($defs.Count -eq $wanted.Count)
foreach ($d in $defs) { . ([scriptblock]::Create($d.Extent.Text)) }

# onnxruntime: переставлять только при необходимости
$p = Get-OnnxPlan @{ "onnxruntime-openvino" = "1.24.1" } "1.24.1"
Check "onnx: nothing to do when only openvino build" (-not $p.remove.Count -and -not $p.reinstall)
$p = Get-OnnxPlan @{ "onnxruntime" = "1.23.2"; "onnxruntime-openvino" = "1.24.1" } "1.24.1"
Check "onnx: plain build removed, openvino reinstalled" (($p.remove -join ",") -eq "onnxruntime" -and $p.reinstall)
$p = Get-OnnxPlan @{ "onnxruntime-directml" = "1.20.0" } "1.24.1"
Check "onnx: directml removed, openvino installed" (($p.remove -join ",") -eq "onnxruntime-directml" -and $p.reinstall)
$p = Get-OnnxPlan @{ "onnxruntime-openvino" = "1.23.0" } "1.24.1"
Check "onnx: wrong openvino version reinstalled" (-not $p.remove.Count -and $p.reinstall)

# токен: без терминала и с -KeepSavedToken вопрос не задаётся
Check "token: given" ((Get-TokenMode "hf_x" $false "" $false) -eq "given")
Check "token: keep saved" ((Get-TokenMode "" $true "hf_saved" $false) -eq "saved")
Check "token: keep without saved" ((Get-TokenMode "" $true "" $false) -eq "none")
Check "token: no terminal keeps saved" ((Get-TokenMode "" $false "hf_saved" $true) -eq "saved")
Check "token: terminal asks" ((Get-TokenMode "" $false "hf_saved" $false) -eq "ask")

# лишние файлы в tools/tests/examples: без терминала не спрашиваем и не удаляем; пустой ответ — «оставить»
Check "extra: no terminal shows and keeps" ((Get-ExtraFilesMode "" $true) -eq "Show")
Check "extra: terminal asks" ((Get-ExtraFilesMode "" $false) -eq "Ask")
Check "extra: explicit mode wins, any case" ((Get-ExtraFilesMode "delete" $true) -eq "Delete" -and (Get-ExtraFilesMode "KEEP" $false) -eq "Keep" -and (Get-ExtraFilesMode "Show" $false) -eq "Show")
Check "extra: empty answer keeps" ((Get-ExtraFilesAnswer "") -eq "Keep" -and (Get-ExtraFilesAnswer "   ") -eq "Keep" -and (Get-ExtraFilesAnswer $null) -eq "Keep")
Check "extra: n and other answers keep" ((Get-ExtraFilesAnswer "n") -eq "Keep" -and (Get-ExtraFilesAnswer "н") -eq "Keep" -and (Get-ExtraFilesAnswer "q") -eq "Keep")
Check "extra: only the whole word y, yes, д, да deletes" ((Get-ExtraFilesAnswer "y") -eq "Delete" -and (Get-ExtraFilesAnswer "Y") -eq "Delete" -and (Get-ExtraFilesAnswer "yes") -eq "Delete" -and (Get-ExtraFilesAnswer "д") -eq "Delete" -and (Get-ExtraFilesAnswer "да") -eq "Delete" -and (Get-ExtraFilesAnswer " Да ") -eq "Delete")
Check "extra: words that only start with y or д keep" ((Get-ExtraFilesAnswer "yellow") -eq "Keep" -and (Get-ExtraFilesAnswer "yep") -eq "Keep" -and (Get-ExtraFilesAnswer "дальше") -eq "Keep" -and (Get-ExtraFilesAnswer "действуй") -eq "Keep" -and (Get-ExtraFilesAnswer "y n") -eq "Keep")

# сравнение списков пакетов
$before = @("numpy==2.3.5", "torch==2.15.0+cpu", "gigaam @ https://example.invalid/main.zip#sha256=aa", "old_pkg==1.0")
$after = @("numpy==2.3.6", "torch==2.15.0+cpu", "gigaam @ https://example.invalid/main.zip#sha256=aa", "new-pkg==2.0")
$c = Compare-PipFreeze $before $after
Check "freeze: no changes" ((Compare-PipFreeze $before $before).Count -eq 0)
Check "freeze: three changes" ($c.Count -eq 3)
Check "freeze: version change" (@($c | Where-Object { $_ -like "изменён: numpy==2.3.5 -> numpy==2.3.6" }).Count -eq 1)
Check "freeze: added and removed" ((@($c | Where-Object { $_ -like "добавлен: new-pkg*" }).Count -eq 1) -and (@($c | Where-Object { $_ -like "удалён: old_pkg*" }).Count -eq 1))

# процессы: задачи отделяются от сервера
$s = Split-ProgramProcesses @([pscustomobject]@{ Name = "python.exe"; CommandLine = "python C:\p\worker.py job" },
                              [pscustomobject]@{ Name = "python.exe"; CommandLine = "python C:\p\server.py" })
Check "processes: job and server split" ($s.jobs.Count -eq 1 -and $s.other.Count -eq 1)
$s = Split-ProgramProcesses $null
Check "processes: none running" ($s.jobs.Count -eq 0 -and $s.other.Count -eq 0)

# версия программы
$vdir = Join-Path $base "ver"; New-Item -ItemType Directory $vdir | Out-Null
Set-Content (Join-Path $vdir "server.py") 'VERSION = "2099.1.2.3"   # synthetic' -Encoding UTF8
Check "version: read" ((Get-ServerVersion $vdir) -eq "2099.1.2.3")
Check "version: missing" ((Get-ServerVersion (Join-Path $base "none")) -eq "")

# источник: Git-клон
$plain = Join-Path $base "plain"; New-Item -ItemType Directory $plain | Out-Null
$st = Get-SourceState $plain
Check "source: folder without git is accepted" (-not $st.git -and $st.reviewed)
$hasGit = [bool](Get-Command git -ErrorAction SilentlyContinue)
function GitQuiet([string]$repo) { & git -C $repo -c user.name=synthetic -c user.email=synthetic@example.invalid -c core.hooksPath= -c commit.gpgsign=false -c core.autocrlf=false @args 2>&1 | Out-Null }
function New-Repo([string]$name) {
    $r = Join-Path $base $name; New-Item -ItemType Directory $r | Out-Null
    & git init -q $r 2>&1 | Out-Null
    Copy-Item $Script (Join-Path $r "install.ps1"); Copy-Item (Join-Path $vdir "server.py") (Join-Path $r "server.py")
    GitQuiet $r add -A; GitQuiet $r commit -q -m synthetic
    GitQuiet $r update-ref refs/remotes/origin/main HEAD
    return $r
}
if ($hasGit) {
    $repo = New-Repo "repo"
    $st = Get-SourceState $repo
    Check "source: commit in origin/main is reviewed" ($st.git -and $st.onMain -and $st.reviewed -and $st.commit)
    Add-Content (Join-Path $repo "server.py") "# local change"
    $st = Get-SourceState $repo
    Check "source: uncommitted change is not reviewed" ($st.dirty.Count -eq 1 -and -not $st.reviewed)
    GitQuiet $repo commit -q -am local
    $st = Get-SourceState $repo
    Check "source: commit outside origin/main is not reviewed" (-not $st.onMain -and -not $st.reviewed)
    GitQuiet $repo tag v-synthetic
    $st = Get-SourceState $repo
    Check "source: tagged commit is reviewed" ($st.tag -eq "v-synthetic" -and $st.reviewed)
} else { Write-Output "git не найден: проверки клона пропущены" }

# ---------- запуск установщика (без терминала, как у ИИ-агента) ----------
function New-Install([string]$tag) {
    # нелатинская папка: если проверки перед обновлением пройдены, шаг 1 остановит установку
    $t = Join-Path $base "программа-$tag"; $dd = Join-Path $base "data-$tag"
    New-Item -ItemType Directory $t, $dd | Out-Null
    Set-Content (Join-Path $t "server.py") 'VERSION = "2000.1.1.1"' -Encoding UTF8; Set-Content (Join-Path $t "core.py") "x"
    return @{ t = $t; dd = $dd }
}
$src = Join-Path $base "src"; New-Item -ItemType Directory $src | Out-Null
Copy-Item $Script (Join-Path $src "install.ps1"); Copy-Item (Join-Path $vdir "server.py") (Join-Path $src "server.py")
function Run($inst, [string]$scriptPath, [string[]]$extra) {
    $a = @{ Target = $inst.t; DataDir = $inst.dd; SkipClaudeConfig = $true }
    foreach ($x in $extra) { $a[$x.TrimStart("-")] = $true }
    $o = "" | & $scriptPath @a *>&1 | Out-String
    return @{ code = $LASTEXITCODE; out = ($o -replace '\s+', '') }
}
function Untouched($inst) { ((Get-Content (Join-Path $inst.t "core.py") -Raw).Trim() -eq "x") -and -not (Test-Path (Join-Path $inst.t "install.ps1")) }

# J: идёт расшифровка — отказ до шага 1, даже с -AllowRunningServer
$i = New-Install "j"
$global:FakeProcs = @([pscustomobject]@{ ProcessId = 99998; Name = "python.exe"; ExecutablePath = "C:\Python\python.exe"; CommandLine = "python $($i.t)\worker.py job-1" })
$r = Run $i (Join-Path $src "install.ps1") @("-AllowRunningServer")
Check "J job running: exit 1" ($r.code -eq 1)
Check "J job running: stopped before step 1" ($r.out -match "Обновлениененачато:идётобработка" -and $r.out -notmatch "Проверкаместа")
Check "J job running: versions shown" ($r.out -match "2099\.1\.2\.3" -and $r.out -match "2000\.1\.1\.1")
Check "J target untouched" (Untouched $i)

# K: работает только сервер — без флага отказ с подсказкой
$i = New-Install "k"
$global:FakeProcs = @([pscustomobject]@{ ProcessId = 99997; Name = "python.exe"; ExecutablePath = "C:\Python\python.exe"; CommandLine = "python $($i.t)\server.py" })
$r = Run $i (Join-Path $src "install.ps1") @()
Check "K server running: exit 1 with hint" ($r.code -eq 1 -and $r.out -match "AllowRunningServer" -and $r.out -notmatch "Проверкаместа")
Check "K target untouched" (Untouched $i)

# L: работает только сервер, -AllowRunningServer — проверки пройдены, дальше шаг 1
$r = Run $i (Join-Path $src "install.ps1") @("-AllowRunningServer")
Check "L server allowed: reaches step 1" ($r.out -match "Проверкаместа" -and $r.out -match "кириллица")
Check "L target untouched" (Untouched $i)
$global:FakeProcs = @()

# M: -Token вместе с -KeepSavedToken — отказ
$i = New-Install "m"
$o = "" | & (Join-Path $src "install.ps1") -Target $i.t -DataDir $i.dd -SkipClaudeConfig -Token "hf_synthetic" -KeepSavedToken *>&1 | Out-String
Check "M token and keep: exit 1" ($LASTEXITCODE -eq 1 -and ($o -replace '\s+', '') -match "-Tokenили-KeepSavedToken")

# N, O: непроверенный источник
if ($hasGit) {
    $dirty = New-Repo "dirty-src"; Add-Content (Join-Path $dirty "server.py") "# local change"
    $i = New-Install "n"
    $r = Run $i (Join-Path $dirty "install.ps1") @()
    Check "N unreviewed source: exit 1 before step 1" ($r.code -eq 1 -and $r.out -match "AllowUnreviewedSource" -and $r.out -notmatch "Проверкаместа")
    Check "N target untouched" (Untouched $i)
    $r = Run $i (Join-Path $dirty "install.ps1") @("-AllowUnreviewedSource")
    Check "O unreviewed source allowed: reaches step 1" ($r.out -match "Проверкаместа")
    if ($r.out -notmatch "Проверкаместа") { $global:Diag = "---- O: code=$($r.code) ----`n" + $r.out.Substring([Math]::Max(0, $r.out.Length - 1500)) }
    Check "O target untouched" (Untouched $i)
}

$results | Format-Table -AutoSize | Out-String -Width 200
if ($global:Diag) { $global:Diag }
# файлы .git только для чтения: Remove-Item -Force снимает этот признак (ссылок в этой папке нет)
if (Test-Path -LiteralPath $base) { Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $base) { "не удалось удалить временную папку: $base" }
$bad = @($results | Where-Object { -not $_.ok }).Count
"FAILED: $bad"
if ($bad) { exit 1 }
exit 0
