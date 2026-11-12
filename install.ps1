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

  powershell -ExecutionPolicy Bypass -File .\install.ps1 -Uninstall
      убирает сервер из настроек Claude; отдельно спрашивает: удалить папку программы с моделями и кэшами
      (по умолчанию да) и удалить личные данные (по умолчанию НЕТ)

Повторный запуск безопасен: обновляет программу, не трогая данные пользователя (папка данных: профили, словари, config.json с результатом теста скорости).
#>
param(
    [string]$Target = "",
    [string]$DataDir = "",
    [string]$Token = "",
    [switch]$SkipClaudeConfig,
    [switch]$Uninstall
)
$ErrorActionPreference = "Continue"
# Windows PowerShell 5.1: без этого ConvertTo-Json пишет массивы как {"value":[...],"Count":n}
Remove-TypeData System.Array -ErrorAction SilentlyContinue
[Net.ServicePointManager]::SecurityProtocol = [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
$Source = $PSScriptRoot
$Name = "rus-transcriber"

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }
function Check($what) { if ($LASTEXITCODE -ne 0) { Fail "Ошибка на шаге: $what (код $LASTEXITCODE)" } }
function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function HasNonAscii($s) { return $s -match '[^\x00-\x7F]' }
function WriteUtf8NoBom($path, $text) { [IO.File]::WriteAllText($path, $text, (New-Object Text.UTF8Encoding $false)) }

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

# ---------- где лежит конфиг Claude Desktop (обычная установка или из Microsoft Store) ----------
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
$stray = Join-Path $env:LOCALAPPDATA "Claude-Data\claude_desktop_config.json"
if (Test-Path $stray) {
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
# папка данных: параметр -DataDir -> уже записанная в настройках Claude -> %APPDATA%\rus-transcriber-data
if (-not $DataDir) { $DataDir = Join-Path $env:APPDATA "$Name-data" }
$DataDir = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($DataDir).TrimEnd("\")
if ($DataDir -eq $Target -or $DataDir.StartsWith($Target + "\", [StringComparison]::OrdinalIgnoreCase)) {
    Fail "Папка данных ($DataDir) не может лежать внутри папки программы ($Target): при обновлении она бы потерялась. Укажите другую через -DataDir."
}

# ================================ УДАЛЕНИЕ ================================
if ($Uninstall) {
    Write-Host "Удаление $Name из: $Target" -ForegroundColor Yellow
    foreach ($cp in (Find-ClaudeConfig)) {
        if (Test-Path $cp) {
            Edit-ClaudeConfig $cp { param($s) $s.PSObject.Properties.Remove($Name) }
            Write-Host "  убран из настроек Claude: $cp"
        }
    }
    if (Test-Path (Join-Path $Target "server.py")) {
        $ans = Read-Host "Удалить программу $Target со всеми моделями и кэшами? Данные и расшифровки рядом с аудио не затрагиваются. (Y/n)"
        if (-not $ans -or $ans -match '^(y|д)') { Remove-Item $Target -Recurse -Force; Write-Host "  программа удалена" }
        else { Write-Host "  программа оставлена" }
    }
    if (Test-Path $DataDir) {
        Write-Host "Личные данные (голосовые профили, словари групп, настройки, задачи): $DataDir" -ForegroundColor Yellow
        $ans = Read-Host "Удалить и их? Это нельзя отменить; расшифровки рядом с аудио не затрагиваются. (y/N)"
        if ($ans -match '^(y|д)') { Remove-Item $DataDir -Recurse -Force; Write-Host "  данные удалены" }
        else { Write-Host "  данные оставлены нетронутыми: $DataDir" }
    }
    Write-Host "Python и ffmpeg не удалялись (ими могут пользоваться другие программы)"
    Write-Host "Перезапустите Claude Desktop."
    exit 0
}

# ================================ УСТАНОВКА ================================
Write-Host "Установка $Name в: $Target" -ForegroundColor Green

Step "1/9" "Проверка места установки"
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
$insideProfile = $Target.StartsWith($env:USERPROFILE + "\", [StringComparison]::OrdinalIgnoreCase)
$drive = (Split-Path -Qualifier $Target) + "\"
$free = (Get-PSDrive ($drive.Substring(0,1))).Free / 1GB
Write-Host ("  свободно на {0} {1:N1} ГБ (нужно около 8)" -f $drive, $free)
if ($free -lt 8) { Fail "Мало места. Освободите диск или укажите -Target на другом диске." }

Step "2/9" "Поиск установленного Python (подходят 3.11, 3.12, 3.13)"
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

Step "3/9" "Папка программы и права доступа"
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

Step "4/9" "Копирование файлов программы"
if ($Source -ne $Target) {
    foreach ($f in @("server.py", "worker.py", "bench.py", "backends.py", "paths.py", "transcript.py",
                     "diar_fast.py", "vocab.py", "voices.py", "gender.py", "words.py", "quotes.py", "conspect.py",
                     "review_pack.py", "freeform.py", "jobslots.py", "settings.py", "migrate.py", "data_io.py", "core.py", "confirm.py", "tmpfiles.py", "selfcheck.py", "rt.py", "voice_io.py", "voice_review.py", "review_page.py", "cards.html", "README.md", "install.ps1")) {
        Copy-Item (Join-Path $Source $f) (Join-Path $Target $f) -Force
    }
    $tl = Join-Path $Source "tools"      # проверка качества (eval_quality.py) и проверка репозитория (check_repo.py)
    if (Test-Path $tl) { Copy-Item $tl (Join-Path $Target "tools") -Recurse -Force -Exclude "__pycache__" }
    $ts = Join-Path $Source "tests"      # тесты — для самопроверки (self_check) на этом компьютере
    if (Test-Path $ts) { Copy-Item $ts (Join-Path $Target "tests") -Recurse -Force -Exclude "__pycache__" }
    $ex = Join-Path $Source "examples"
    if (Test-Path $ex) { Copy-Item $ex (Join-Path $Target "examples") -Recurse -Force }   # пример группы (вымышленный)
    # данные (профили, словари, настройки, задачи) лежат в папке данных, установщик их не копирует и не трогает;
    # старые profiles\, config.json, jobs\ из папки программы перенесёт migrate.py при первом запуске сервера
} else { Write-Host "  запуск из папки программы, копировать нечего" }
Set-Location $Target

# все кэши — внутри папки программы
$env:PIP_CACHE_DIR = Join-Path $Target "cache\pip"
$env:HF_HOME       = Join-Path $Target "cache\huggingface"
$env:TORCH_HOME    = Join-Path $Target "cache\torch"

Step "5/9" "Виртуальное окружение"
$py = Join-Path $Target ".venv\Scripts\python.exe"
if (Test-Path $py) {
    $venvVer = (& $py -c "import sys; print('%d.%d' % sys.version_info[:2])").Trim()
    Write-Host "  окружение уже есть (Python $venvVer), обновляю библиотеки"
} else {
    & $basePyExe -m venv (Join-Path $Target ".venv"); Check "venv"
}
& $py -m pip install --upgrade pip wheel; Check "pip"

Step "6/9" "PyTorch (процессор), pyannote, GigaAM, MCP"
& $py -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu; Check "PyTorch"
& $py -m pip install "pyannote.audio>=4.0" "pytorch-metric-learning==2.9.0" soundfile truststore "mcp[cli]" "gigaam @ https://github.com/salute-developers/GigaAM/archive/refs/heads/main.zip"; Check "pyannote/GigaAM"
# pytorch-metric-learning 3.x требует numpy>=2.5.3, а OpenVINO 2025.4.1 — numpy<2.4.0: держим совместимую 2.9.0

Step "7/9" "OpenVINO для видеокарты Arc"
& $py -m pip install "onnx-asr[hub]" "openvino==2025.4.1"; Check "onnx-asr/OpenVINO"
# onnxruntime, onnxruntime-directml и onnxruntime-openvino дают один и тот же модуль:
# убираем обычные сборки и ставим сборку с OpenVINO последней
& $py -m pip uninstall -y onnxruntime onnxruntime-directml onnxruntime-gpu
& $py -m pip install --force-reinstall --no-deps "onnxruntime-openvino==1.24.1"; Check "onnxruntime-openvino"
# под Windows модуль OpenVINO в onnxruntime берёт библиотеки из пакета openvino — версии должны совпадать
# (onnxruntime-openvino 1.24.x собран с OpenVINO 2025.4.1; с 2026.x — «ошибка 127» при загрузке)
# проверяем именно версию: одной проверки «import openvino» мало, потому что любая версия импортируется
$ov = & $py -c "import openvino; print(openvino.__version__.split('-')[0])" 2>$null
if ($ov -ne "2025.4.1") {
    Write-Host "  OpenVINO $ov не подходит — ставлю 2025.4.1"
    & $py -m pip install "openvino==2025.4.1"; Check "OpenVINO 2025.4.1"
} else { Write-Host "  OpenVINO 2025.4.1 — версия подходит" }
# после смены версии OpenVINO сохранённые модели Whisper пересоберутся при первом запуске (это нормально)
# словарь русского языка: свободные ответы в карточках — отличать опечатку от настоящего слова
& $py -m pip install pymorphy3 pymorphy3-dicts-ru; Check "pymorphy3"
& $py -c "import paths, backends as b; print('  OpenVINO видит:', b.openvino_devices()); print('  ONNX Runtime:', b.available_providers())"
Write-Host "  нужны CPU, OpenVINOExecutionProvider и желательно GPU; NPU не обязателен (на Meteor Lake он для GigaAM не используется)" -ForegroundColor DarkGray

Step "8/9" "Токен Hugging Face"
if (-not $Token) {
    $sec = Read-Host "Вставьте токен Hugging Face (hf_..., тип Read). Enter — пропустить" -AsSecureString
    $Token = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($sec))
}
$Token = $Token.Trim()
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
} else {
    Write-Host "  пропущено: допишите HF_TOKEN в настройки Claude позже (см. README)" -ForegroundColor Yellow
}

Step "9/9" "Подключение к Claude Desktop"
if ($SkipClaudeConfig) {
    Write-Host "  пропущено (-SkipClaudeConfig)"
} else {
    # Claude Desktop при выходе сохраняет настройки поверх файла и затирает нашу запись,
    # поэтому писать можно только при полностью закрытом приложении
    while (Get-Process -Name "Claude" -ErrorAction SilentlyContinue) {
        Write-Host "  Claude Desktop запущен. Закройте его полностью: правой кнопкой по значку Claude в трее → Quit (Выход)." -ForegroundColor Yellow
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
    Write-Host "  Теперь запустите Claude Desktop." -ForegroundColor Yellow
}

# старые кэши от первой версии в профиле можно удалить
$old = @("$env:USERPROFILE\.cache\gigaam", "$env:USERPROFILE\.cache\huggingface") | Where-Object { Test-Path $_ }
if ($old) {
    Write-Host "`nНайдены кэши прошлой установки (их можно удалить, если они не нужны другим программам):" -ForegroundColor DarkGray
    $old | ForEach-Object { Write-Host "  $_" -ForegroundColor DarkGray }
}

Write-Host "`nГотово. Программа, модели и кэши: $Target" -ForegroundColor Green
Write-Host "Личные данные: $DataDir (установщик их не трогает; перенос между компьютерами — export_group / import_group)" -ForegroundColor Green
Write-Host "Дальше — тест скорости. Попросите Claude: «запусти тест скорости на C:\...\запись.mp3»,"
Write-Host "или вручную:  `$env:HF_TOKEN='hf_...'; & '$py' '$Target\bench.py' 'C:\путь\к\записи.mp3'"
