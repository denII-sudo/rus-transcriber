# Синтетическая проверка удаления устаревших файлов при обновлении (запускает tests/test_install_stale.py).
# install.ps1 запускается в режиме -AcceptanceCopyOnly внутри временной папки с файлом-меткой: копируются
# только файлы программы (шаг 4), без Python-окружения, сети и настоящих данных. Источник — синтетический,
# с копией install.ps1 и buildinfo.py из этого репозитория; настоящая установка и папка данных не затрагиваются.
param([string]$Script)
$ErrorActionPreference = "Continue"
$results = @()
$global:LastRun = $null; $global:Diags = @()
function Check($name, $cond) {
    $script:results += [pscustomobject]@{ test = $name; ok = [bool]$cond }
    # при провале — хвост вывода последнего запуска установщика (для разбора в журнале CI)
    if (-not $cond -and $global:LastRun) { $global:Diags += "---- $name (код $($global:LastRun.code)) ----`n" + $global:LastRun.text.Substring([Math]::Max(0, $global:LastRun.text.Length - 1800)) }
}

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    "python не найден: сценарии пропущены"
    "FAILED: 0"
    exit 0
}
$repo = Split-Path $Script -Parent
# временная папка на весь прогон; переменные окружения указывают внутрь неё
$base = Join-Path ([IO.Path]::GetTempPath()) ("stale-" + [guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory $base | Out-Null
$env:LOCALAPPDATA = Join-Path $base "local"; $env:APPDATA = Join-Path $base "roaming"; $env:USERPROFILE = $base
$env:TEMP = Join-Path $base "temp"; $env:TMP = $env:TEMP
foreach ($d in @($env:LOCALAPPDATA, $env:APPDATA, $env:TEMP)) { New-Item -ItemType Directory $d -Force | Out-Null }
$env:RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX = "1"
$sb = Join-Path $base "sandbox"; New-Item -ItemType Directory $sb | Out-Null
Set-Content (Join-Path $sb "rus-transcriber-acceptance.sandbox") "synthetic acceptance sandbox"
$prog = Join-Path $sb "program"; $data = Join-Path $sb "data"
New-Item -ItemType Directory $prog, $data | Out-Null

function Put([string]$root, [string]$rel, [string]$text) {
    $p = Join-Path $root $rel
    New-Item -ItemType Directory -Force (Split-Path $p) | Out-Null
    [IO.File]::WriteAllText($p, $text, (New-Object Text.UTF8Encoding $false))
}
function At([string]$rel) { Join-Path $prog $rel }
function Has([string]$rel) { Test-Path -LiteralPath (At $rel) }
# точное имя (с учётом регистра): Test-Path регистр не различает
function Exact([string]$rel) { $leaf = Split-Path $rel -Leaf; (Test-Path -LiteralPath (At $rel)) -and ((@(Get-ChildItem -LiteralPath (Split-Path (At $rel) -Parent) -Force -Name)) -ccontains $leaf) }
$docs = @("model-download-certificate.md", "2026-11-17-p3-workflow-handoff.md",
          "2026-11-17-p4-opencode-openai-handoff.md", "2026-11-17-p5-screen-handoff.md")
function New-Source([string]$name, [string[]]$files) {
    $d = Join-Path $base $name
    New-Item -ItemType Directory $d | Out-Null
    Copy-Item $Script (Join-Path $d "install.ps1")
    Copy-Item (Join-Path $repo "buildinfo.py") (Join-Path $d "buildinfo.py")
    Put $d "server.py" 'VERSION = "2099.1.1.1"'
    foreach ($doc in $docs) { Put $d "docs\$doc" "synthetic doc" }
    foreach ($f in $files) { Put $d $f "# synthetic $f" }
    return $d
}
function Install([string]$srcDir, [hashtable]$more = @{}) {
    $a = @{ Target = $prog; DataDir = $data; SkipClaudeConfig = $true; AcceptanceCopyOnly = $true }
    foreach ($k in $more.Keys) { $a[$k] = $more[$k] }
    Push-Location $base
    $o = "" | & (Join-Path $srcDir "install.ps1") @a *>&1 | Out-String
    $code = $LASTEXITCODE
    Pop-Location
    $global:LastRun = @{ code = $code; text = $o }
    return @{ code = $code; text = $o; flat = ($o -replace '\s+', '') }
}
function Manifest-Files { (Get-Content (At "build-info.json") -Raw -Encoding UTF8 | ConvertFrom-Json).files.PSObject.Properties.Name }

# ---------- A: первая установка прежней сборки ----------
$v1 = New-Source "v1" @("old_mod.py", "cards.html", "tools/check.py", "tools/gone.py", "tests/test_old.py",
                        "tests/test_changed_old.py", "tests/sub/nested_old.py", "tests/test_keep.py", "examples/e.json")
Put $data "user-sentinel.txt" "preserve"
$r = Install $v1
Check "A first install: exit 0" ($r.code -eq 0)
Check "A first install: files and manifest" ((Has "old_mod.py") -and (Has "tools/gone.py") -and (Has "build-info.json") -and ((Manifest-Files) -contains "old_mod.py"))
Check "A first install: nothing to remove, no question" ($r.flat -notmatch "будутудалены" -and $r.flat -notmatch "Удалитьэтифайлы")

# файлы человека и правка прежнего файла комплекта
Put $prog "tools/mine.txt" "mine"
Put $prog "mine_script.py" "# my own script in the program folder"
Set-Content (At "tests/test_changed_old.py") "# edited by the user"

# ---------- B: обновление без параметра и без терминала ----------
$v2 = New-Source "v2" @("cards.html", "tools/check.py", "tools/new_tool.py", "tests/test_keep.py", "examples/e.json")
$noTerminal = [Console]::IsInputRedirected
$r = Install $v2
Check "B update: exit 0, no hang" ($r.code -eq 0)
Check "B stale root file of previous build removed" (-not (Has "old_mod.py"))
Check "B stale tools and tests files removed" (-not (Has "tools/gone.py") -and -not (Has "tests/test_old.py") -and -not (Has "tests/sub/nested_old.py"))
Check "B emptied subfolder removed, folders kept" (-not (Has "tests/sub") -and (Has "tests") -and (Has "tools") -and (Has "docs"))
Check "B new and kept files in place" ((Has "tools/new_tool.py") -and (Has "tools/check.py") -and (Has "tests/test_keep.py") -and (Has "cards.html") -and (Has "server.py"))
Check "B file edited by the user is kept" ((Has "tests/test_changed_old.py") -and ((Get-Content (At "tests/test_changed_old.py") -Raw) -match "edited by the user"))
Check "B user files kept (default does not delete)" ((Has "tools/mine.txt") -and (Has "mine_script.py"))
Check "B removed files are listed with the way to keep them" ($r.flat -match "-KeepStaleFiles" -and $r.flat -match "old_mod\.py" -and $r.flat -match "tools/gone\.py" -and $r.flat -match "tests/test_old\.py")
Check "B kept files are listed with a hint" ($r.flat -match "mine\.txt" -and $r.flat -match "mine_script\.py" -and $r.flat -match "-ExtraFilesDelete")
Check "B manifest describes the new build" (-not ((Manifest-Files) -contains "old_mod.py") -and ((Manifest-Files) -contains "tools/new_tool.py"))
Check "B data folder untouched" ((Get-Content (Join-Path $data "user-sentinel.txt") -Raw).Trim() -eq "preserve")
if (-not $noTerminal) { "ввод не перенаправлен: сценарий B выполнен в терминале" }

# ---------- C: -ExtraFiles Delete ----------
$r = Install $v2 @{ ExtraFiles = "Delete" }
Check "C delete mode: exit 0" ($r.code -eq 0)
Check "C delete mode: files outside the kit in tools/tests/examples removed" (-not (Has "tools/mine.txt") -and -not (Has "tests/test_changed_old.py"))
Check "C delete mode: kit and root files stay" ((Has "tools/check.py") -and (Has "tests/test_keep.py") -and (Has "mine_script.py") -and (Has "server.py"))
Check "C delete mode: removed files are listed" ($r.flat -match "mine\.txt")

# ---------- D: -ExtraFiles Keep ----------
Put $prog "tools/mine2.txt" "mine"
$r = Install $v2 @{ ExtraFiles = "Keep" }
Check "D keep mode: kept without a list" ((Has "tools/mine2.txt") -and $r.flat -notmatch "mine2\.txt" -and $r.code -eq 0)

# ---------- E: прежнего build-info.json нет ----------
Remove-Item (At "build-info.json") -Force
Put $prog "tools/gone2.py" "# was installed by an older build, nothing proves it"
Put $prog "old2_mod.py" "# same, root"
$r = Install $v2 @{ ExtraFiles = "Show" }
Check "E no manifest: nothing deleted" ((Has "tools/gone2.py") -and (Has "tools/mine2.txt") -and (Has "old2_mod.py") -and $r.code -eq 0)
Check "E no manifest: files are shown" ($r.flat -match "gone2\.py" -and $r.flat -match "old2_mod\.py" -and $r.flat -match "build-info\.json")
Check "E manifest written again" (Has "build-info.json")
Remove-Item (At "tools/gone2.py"), (At "old2_mod.py"), (At "tools/mine2.txt") -Force

# ---------- F: ссылка (junction) внутри папки программы ----------
$outside = Join-Path $base "outside"
Put $outside "victim.py" "# synthetic victim"
$v3 = New-Source "v3" @("cards.html", "tools/check.py", "tools/jn/victim.py")
$r = Install $v3
Check "F setup: kit with a nested tools folder installed" ($r.code -eq 0 -and (Has "tools/jn/victim.py"))
Remove-Item (At "tools/jn") -Recurse -Force
New-Item -ItemType Junction -Path (At "tools/jn") -Target $outside | Out-Null
$jn = Get-Item -LiteralPath (At "tools/jn") -Force
Check "F setup: junction created" ([bool]($jn.Attributes -band [IO.FileAttributes]::ReparsePoint))
$v4 = New-Source "v4" @("cards.html", "tools/check.py", "tools/v4_only.py", "tools/jn/victim.py")
$r = Install $v4 @{ ExtraFiles = "Delete" }
# ссылка внутри tools: установка отказывает до копирования, файл за ссылкой не перезаписан и не удалён
Check "F junction: refused before any change" ($r.code -eq 1 -and $r.flat -match "ссылки" -and $r.flat -match "tools/jn")
Check "F junction: file behind the link is intact, kit not written through it" ((Get-Content (Join-Path $outside "victim.py") -Raw).Trim() -eq "# synthetic victim")
Check "F junction: nothing of the new kit copied" (-not (Has "tools/v4_only.py"))
Check "F junction: the link itself is not touched" (Test-Path -LiteralPath (At "tools/jn"))
[IO.Directory]::Delete((At "tools/jn"))
Check "F junction removed without touching its target" (Test-Path -LiteralPath (Join-Path $outside "victim.py"))

# ---------- G: в новой версии нет папки tools ----------
$v5 = New-Source "v5" @("cards.html")
$r = Install $v5
Check "G source without tools: installed tools kept" ((Has "tools/check.py") -and $r.code -eq 0)

# ---------- H: неверное значение параметра ----------
$r = Install $v2 @{ ExtraFiles = "Purge" }
Check "H wrong -ExtraFiles value: refused before any change" ($r.code -eq 1 -and $r.flat -match "ExtraFilesпринимает")

# ---------- I: переименование только регистра, файл с точкой, сбой копирования ----------
Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$v6 = New-Source "v6" @("cards.html", "tools/Foo.py", "tools/check.py", "gone_late.py", "tools/gone_tool.py")
$r = Install $v6
Check "I setup: install" ($r.code -eq 0 -and (Has "gone_late.py") -and (Has "tools/Foo.py"))
# новая версия: tools/Foo.py -> tools/foo.py (содержимое то же), в комплекте файл с точкой, устаревшие файлы убраны из комплекта
$v7 = New-Source "v7" @("cards.html", "tools/check.py")
Put $v7 "tools/foo.py" "# synthetic tools/Foo.py"
Put $v7 "tools/.keep" "shipped dot-file"
Put $prog "tools/mine.txt" "mine"
$r = Install $v7 @{ ExtraFiles = "Delete" }
Check "I update: exit 0" ($r.code -eq 0)
Check "I case-only rename: the installed file is not removed" (Has "tools/Foo.py")
Check "I stale files of the previous build removed" (-not (Has "gone_late.py") -and -not (Has "tools/gone_tool.py"))
Check "I dot-file of the kit survives -ExtraFiles Delete" (Has "tools/.keep")
Check "I user file shown before the copy is deleted, kit dot-file reported as not shown" (-not (Has "tools/mine.txt") -and $r.flat -match "неподказывали|непоказывали")

# сбой копирования (файл назначения заперт): удаление не выполняется
Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$v8 = New-Source "v8" @("cards.html", "tools/check.py", "gone_late.py", "tools/gone_tool.py")
$r = Install $v8
Check "I2 setup: install" ($r.code -eq 0 -and (Has "gone_late.py"))
$v9 = New-Source "v9" @("cards.html", "tools/check.py")
Put $v9 "tools/check.py" "# changed content so that the copy has to overwrite the locked file"
$lock = [IO.File]::Open((At "tools/check.py"), [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
try { $r = Install $v9 } finally { $lock.Dispose() }
Check "I2 locked file: copy error reported" ($r.flat -match "неудалосьскопировать")
Check "I2 locked file: nothing is deleted after a failed copy" ((Has "gone_late.py") -and (Has "tools/gone_tool.py"))
Check "I2 locked file: deletion skipped message" ($r.flat -match "копированиепрошлосошибками")

# ---------- K: -KeepStaleFiles ----------
Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$v10 = New-Source "v10" @("cards.html", "tools/check.py", "stale_root.py", "tools/stale_tool.py", "tests/test_stale.py")
$r = Install $v10
Check "K setup: install" ($r.code -eq 0 -and (Has "stale_root.py"))
$v11 = New-Source "v11" @("cards.html", "tools/check.py")
$r = Install $v11 @{ KeepStaleFiles = $true; ExtraFiles = "Delete" }
Check "K keep stale: exit 0" ($r.code -eq 0)
Check "K keep stale: nothing of the previous build is deleted" ((Has "stale_root.py") -and (Has "tools/stale_tool.py") -and (Has "tests/test_stale.py"))
Check "K keep stale: files are listed as kept" ($r.flat -match "stale_root\.py" -and $r.flat -match "tools/stale_tool\.py" -and $r.flat -match "оставлены\(-KeepStaleFiles\)")
Check "K keep stale: no deletion announced" ($r.flat -notmatch "будутудалены")

# ---------- K2: оставленные -KeepStaleFiles остаются в build-info.json и предлагаются снова ----------
Check "K2 kept stale files stay in the manifest" (((Manifest-Files) -contains "stale_root.py") -and ((Manifest-Files) -contains "tools/stale_tool.py"))
$r = Install $v11
Check "K2 next update without the parameter: kept files are offered and removed (tests/ is not in the source: left alone)" ($r.code -eq 0 -and -not (Has "stale_root.py") -and -not (Has "tools/stale_tool.py") -and (Has "tests/test_stale.py"))
Check "K2 removed files leave the manifest" (-not ((Manifest-Files) -contains "stale_root.py"))

# ---------- L: консоль с кодовой страницей 65001 (в Windows PowerShell 5.1 конвейер в программу начинается с BOM) ----------
# $OutputEncoding с преамбулой даёт тот же BOM в начале ввода внешней программы, что и chcp 65001
Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$v12 = New-Source "v12" @("cards.html", "tools/check.py", "l_stale.py", "tools/l_stale_tool.py")
$r = Install $v12
Check "L setup: install" ($r.code -eq 0 -and (Has "l_stale.py"))
$v13 = New-Source "v13" @("cards.html", "tools/check.py")
$savedEncoding = $OutputEncoding
$OutputEncoding = New-Object System.Text.UTF8Encoding $true
try { $r = Install $v13 @{ ExtraFiles = "Delete" } } finally { $OutputEncoding = $savedEncoding }
Check "L BOM in piped input: stale files are still removed" ($r.code -eq 0 -and -not (Has "l_stale.py") -and -not (Has "tools/l_stale_tool.py"))
Check "L BOM in piped input: no error shown" ($r.flat -notmatch "неудалосьудалить" -and $r.flat -notmatch "couldnotbelisted")
$bomFile = Join-Path $base "bom-shown.json"
[IO.File]::WriteAllBytes($bomFile, [byte[]](0xEF, 0xBB, 0xBF) + [Text.Encoding]::ASCII.GetBytes('{"stale":[]}'))
$o = & python -B (Join-Path $v13 "buildinfo.py") --source $v13 --target $prog --prune stale --shown-file $bomFile
Check "L shown list with a BOM is read" ($LASTEXITCODE -eq 0 -and (($o -join "") -match '"removed": \[\]'))

# ---------- M: переименование только регистра, точные имена ----------
Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$m1 = New-Source "m1" @("cards.html", "tools/Foo.py", "tests/Case_Test.py")
Put $m1 "Mod.py" "# synthetic mod"
$r = Install $m1
Check "M setup: install" ($r.code -eq 0 -and (Exact "Mod.py") -and (Exact "tools/Foo.py"))
$m2 = New-Source "m2" @("cards.html")
Put $m2 "mod.py" "# synthetic mod"
Put $m2 "tools/foo.py" "# synthetic tools/Foo.py"
Put $m2 "tests/Case_Test.py" "# synthetic tests/Case_Test.py"
$r = Install $m2
Check "M case-only rename: exit 0" ($r.code -eq 0)
Check "M root file has the new exact name" ((Exact "mod.py") -and -not (Exact "Mod.py"))
Check "M tools file has the new exact name" ((Exact "tools/foo.py") -and -not (Exact "tools/Foo.py"))
Check "M manifest holds the new names" (((Manifest-Files) -ccontains "mod.py") -and ((Manifest-Files) -ccontains "tools/foo.py"))
$o = & python -B (Join-Path $m2 "buildinfo.py") --source $prog
Check "M build is not shown as changed" (($o -join "") -match '"manifest_status": "matched"')

# ---------- N: ошибка копирования части папки, устаревший файл только для чтения ----------
Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$n1 = New-Source "n1" @("cards.html", "tools/check.py", "gone_late.py", "tools/sub/old_in_sub.py")
$r = Install $n1
Check "N setup: install" ($r.code -eq 0 -and (Has "gone_late.py") -and (Has "tools/sub/old_in_sub.py"))
$n2 = New-Source "n2" @("cards.html", "tools/check.py", "tools/sub/new_in_sub.py", "tools/zz.py")
$denied = At "tools/sub"
& icacls $denied /deny "*S-1-1-0:(OI)(CI)(WD,AD)" /Q | Out-Null
try { $r = Install $n2 } finally { & icacls $denied /remove:d "*S-1-1-0" /Q | Out-Null }
Check "N copy error: exit code 2" ($r.code -eq 2)
Check "N copy error: the rest of the folder is still copied" (Has "tools/zz.py")
Check "N copy error: nothing is deleted" ((Has "gone_late.py") -and (Has "tools/sub/old_in_sub.py") -and $r.flat -match "копированиепрошлосошибками")
Check "N copy error: stale files stay in the manifest" (((Manifest-Files) -contains "gone_late.py") -and ((Manifest-Files) -contains "tools/sub/old_in_sub.py"))
$r = Install $n2
Check "N next run: stale files offered again and removed" ($r.code -eq 0 -and -not (Has "gone_late.py") -and (Has "tools/sub/new_in_sub.py"))

Get-ChildItem $prog -Force | Remove-Item -Recurse -Force
$q1 = New-Source "q1" @("cards.html", "tools/check.py", "ro_stale.py")
$r = Install $q1
Set-ItemProperty -LiteralPath (At "ro_stale.py") -Name IsReadOnly -Value $true
$q2 = New-Source "q2" @("cards.html", "tools/check.py")
$r = Install $q2
Check "N2 read-only stale file: exit code 2, file kept" ($r.code -eq 2 -and (Has "ro_stale.py") -and $r.flat -match "неудалосьудалить")
Check "N2 read-only stale file stays in the manifest" ((Manifest-Files) -contains "ro_stale.py")
Set-ItemProperty -LiteralPath (At "ro_stale.py") -Name IsReadOnly -Value $false
$r = Install $q2
Check "N2 next run: removed, exit 0" ($r.code -eq 0 -and -not (Has "ro_stale.py"))

$results | Format-Table -AutoSize | Out-String -Width 200
$global:Diags | Select-Object -First 4
if ($global:Diag) { $global:Diag }
if (Test-Path -LiteralPath (At "tools/jn")) { try { [IO.Directory]::Delete((At "tools/jn")) } catch { } }
Remove-Item Env:\RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $base) { Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $base) { "не удалось удалить временную папку: $base" }
$bad = @($results | Where-Object { -not $_.ok }).Count
"FAILED: $bad"
if ($bad) { exit 1 }
exit 0
