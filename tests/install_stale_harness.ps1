# Синтетическая проверка удаления устаревших файлов при обновлении (запускает tests/test_install_stale.py).
# install.ps1 запускается в режиме -AcceptanceCopyOnly внутри временной папки с файлом-меткой: копируются
# только файлы программы (шаг 4), без Python-окружения, сети и настоящих данных. Источник — синтетический,
# с копией install.ps1 и buildinfo.py из этого репозитория; настоящая установка и папка данных не затрагиваются.
param([string]$Script)
$ErrorActionPreference = "Continue"
$results = @()
function Check($name, $cond) { $script:results += [pscustomobject]@{ test = $name; ok = [bool]$cond } }

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
Check "B removed files are listed" ($r.flat -match "old_mod\.py" -and $r.flat -match "tools/gone\.py" -and $r.flat -match "tests/test_old\.py")
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

$results | Format-Table -AutoSize | Out-String -Width 200
if ($global:Diag) { $global:Diag }
if (Test-Path -LiteralPath (At "tools/jn")) { try { [IO.Directory]::Delete((At "tools/jn")) } catch { } }
Remove-Item Env:\RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath $base) { Remove-Item -LiteralPath $base -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path -LiteralPath $base) { "не удалось удалить временную папку: $base" }
$bad = @($results | Where-Object { -not $_.ok }).Count
"FAILED: $bad"
if ($bad) { exit 1 }
exit 0
