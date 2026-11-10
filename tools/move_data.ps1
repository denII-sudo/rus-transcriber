<#
Переносит данные rus-transcriber из скрытых папок пакета Claude (Store) в обычную папку, по умолчанию
%APPDATA%\rus-transcriber-data. Запускать при ЗАКРЫТОМ Claude Desktop, обычным PowerShell (не из Claude):
    powershell -ExecutionPolicy Bypass -File .\move_data.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File .\move_data.ps1
    powershell -ExecutionPolicy Bypass -File .\move_data.ps1 -Dest D:\rus-transcriber-data
Копируются только отсутствующие файлы; при расхождении остаётся более новый, второй кладётся в <назначение>\_conflicts\.
Каждый файл проверяется контрольной суммой. Исходная папка затем переименовывается в <имя>.moved-<дата> (не удаляется).
#>
param([string]$Dest = "", [string[]]$Src = @(), [switch]$DryRun, [switch]$KeepSource)
$ErrorActionPreference = "Stop"
$name = "rus-transcriber-data"
if (-not $Dest) { $Dest = Join-Path $env:APPDATA $name }
if ($Dest -match "\\Packages\\") { throw "Назначение $Dest лежит внутри пакета приложения — выберите обычную папку." }
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$destReal = (Resolve-Path $Dest).Path.TrimEnd('\')
$cands = @($Src)
foreach ($pkg in Get-ChildItem (Join-Path $env:LOCALAPPDATA "Packages") -Directory -Filter "Claude_*" -ErrorAction SilentlyContinue) {
    $cands += (Join-Path $pkg.FullName "LocalCache\Local\$name")
    $cands += (Join-Path $pkg.FullName "LocalCache\Roaming\$name")
}
$cands += (Join-Path $env:LOCALAPPDATA $name), (Join-Path $env:APPDATA $name), (Join-Path $env:USERPROFILE $name)
$srcs = @(); $seen = @{}
foreach ($c in $cands) {
    if ($c -and (Test-Path $c -PathType Container)) {
        $r = (Resolve-Path $c).Path.TrimEnd('\')
        if ($r -ne $destReal -and -not $seen.ContainsKey($r)) { $seen[$r] = 1; $srcs += $r }
    }
}
if (-not $srcs) { Write-Host "Источников не найдено: переносить нечего."; return }
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$log = @()
function Say($m) { Write-Host $m; $script:log += ("{0} {1}" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"), $m) }
Say "назначение: $Dest"
foreach ($s in $srcs) {
    $files = Get-ChildItem $s -Recurse -File | Where-Object { $_.Name -notlike ".migrate.lock*" }
    Say "источник: $s ($($files.Count) файлов)"
    if ($DryRun) { continue }
    $copied = 0; $same = 0; $conf = 0
    foreach ($f in $files) {
        $rel = $f.FullName.Substring($s.Length).TrimStart('\')
        $d = Join-Path $Dest $rel
        New-Item -ItemType Directory -Force -Path (Split-Path $d) | Out-Null
        if (Test-Path $d) {
            if ((Get-FileHash $d).Hash -eq (Get-FileHash $f.FullName).Hash) { $same++; continue }
            $dItem = Get-Item $d
            $newerSrc = $f.LastWriteTime -gt $dItem.LastWriteTime
            $cdir = Join-Path $Dest "_conflicts\$stamp\$(Split-Path $rel)"
            New-Item -ItemType Directory -Force -Path $cdir | Out-Null
            if ($newerSrc) { Copy-Item $d (Join-Path $cdir ((Split-Path $rel -Leaf) + ".dest")) -Force; Copy-Item $f.FullName $d -Force }
            else { Copy-Item $f.FullName (Join-Path $cdir ((Split-Path $rel -Leaf) + ".src")) -Force }
            $conf++; Say "расхождение ${rel}: оставлен более новый, второй в _conflicts\$stamp"
            continue
        }
        $tmp = "$d.migrating"
        Copy-Item $f.FullName $tmp -Force
        if ((Get-FileHash $tmp).Hash -ne (Get-FileHash $f.FullName).Hash) { Remove-Item $tmp -Force; throw "Контрольная сумма не совпала: $($f.FullName). Источник не тронут." }
        Move-Item $tmp $d -Force
        $copied++
    }
    foreach ($f in $files) { $rel = $f.FullName.Substring($s.Length).TrimStart('\'); if (-not (Test-Path (Join-Path $Dest $rel))) { throw "В назначении не хватает $rel. Источник не тронут." } }
    Say "скопировано $copied, уже было $same, расхождений $conf"
    if (-not $KeepSource) {
        $new = "$s.moved-$stamp"
        try { Rename-Item $s (Split-Path $new -Leaf); Say "исходная папка переименована: $new" }
        catch { Say "переименовать не удалось ($($_.Exception.Message)) — оставлена как есть" }
    }
}
if ($DryRun) { Write-Host "Пробный прогон: ничего не изменено."; return }
Add-Content -Path (Join-Path $Dest "migration.log") -Value $log -Encoding UTF8
Write-Host "Готово. Запустите Claude Desktop и попросите проверить get_settings."
