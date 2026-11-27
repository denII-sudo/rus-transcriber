# Если модель не скачивается из-за сертификата

Инструкция для случая, когда при первом скачивании моделей GigaAM появляется
ошибка проверки TLS-сертификата:

```
urllib.error.URLError: <urlopen error [SSL: CERTIFICATE_VERIFY_FAILED]
certificate verify failed: certificate has expired (_ssl.c:1010)>
```

или её вариант из хранилища Windows (через `truststore`):

```
urllib.error.URLError: ('Истек/не наступил срок действия требуемого сертификата
при проверке по системным часам или по отметке времени в подписанном файле.',)
```

Это не всегда проблема вашего компьютера. Ниже — как проверить, кто виноват,
и как один раз скачать веса вручную, если сломан сертификат сервера.

## 1. Проверить сертификат сервера

Из папки программы:

```powershell
.\.venv\Scripts\python.exe -c "import paths,ssl,socket; h='cdn.chatwm.opensmodel.sberdevices.ru'; ctx=ssl.create_default_context(); s=socket.create_connection((h,443),timeout=15); ss=ctx.wrap_socket(s,server_hostname=h); print('OK, действует до', ss.getpeercert()['notAfter'])"
```

Если сертификат просрочен, команда завершится ошибкой
`CERTIFICATE_VERIFY_FAILED: certificate has expired`. Посмотреть даты можно так:

Следующая команда получает сертификат без проверки **только для диагностики**,
чтобы показать его издателя и срок. Она ничего не скачивает; не переносите
`_create_unverified_context()` в загрузчик моделей.

```powershell
.\.venv\Scripts\python.exe -c "import ssl,socket,tempfile,os; h='cdn.chatwm.opensmodel.sberdevices.ru'; ctx=ssl._create_unverified_context(); s=socket.create_connection((h,443),timeout=15); ss=ctx.wrap_socket(s,server_hostname=h); f=tempfile.NamedTemporaryFile(delete=False); f.write(ssl.DER_cert_to_PEM_cert(ss.getpeercert(binary_form=True)).encode()); f.close(); c=ssl._ssl._test_decode_cert(f.name); os.unlink(f.name); print(c['subject']); print(c['issuer']); print(c['notBefore']); print(c['notAfter'])"
```

Как читать результат:

| Что видно | Что это значит | Что делать |
|---|---|---|
| `certificate has expired` (код 10) | Истёк листовой сертификат сервера | Ждать продления у владельца сервера или разово скачать вручную (раздел 2) |
| `unable to get local issuer certificate` (код 20) | Клиент не доверяет корню цепочки | Если издатель — `Russian Trusted Sub CA` / НУЦ Минцифры, установить корень в хранилище Windows (раздел 5) |
| `hostname mismatch` | Подменён адрес или прокси | Проверить прокси и антивирус |
| Ошибка и на других сайтах (`pypi.org`, `huggingface.co`) | Проблема локальная | Проверить версию `certifi`, прокси, антивирус |

Быстрая проверка «продлили или нет» без скачивания файла:

```powershell
.\.venv\Scripts\python.exe -c "import paths, urllib.request as u; r=u.urlopen(u.Request('https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/emo.ckpt', headers={'Range':'bytes=0-0'}), timeout=20); print(r.status)"
```

Пока сертификат просрочен, команда падает с `CERTIFICATE_VERIFY_FAILED`.

## 2. Разовая ручная загрузка (только по явному согласию)

**Важно.** HTTPS у сервера сломан, поэтому файлы загружаются по обычному HTTP —
без шифрования канала. Целостность проверяется по md5 из пакета `gigaam`:
подмена или обрыв загрузки будут отсечены. md5 слабее TLS против
целенаправленной атаки, поэтому делайте это только для публичных весов
моделей и только в доверенной сети, пока сертификат сервера не продлят.
Не отключайте проверку TLS в коде и не используйте `--trusted-host` или
`verify=False`.

Готовые команды для вашей установки печатает модуль подсказки:

```powershell
.\.venv\Scripts\python.exe download_hint.py
```

То же самое вручную (путь `$dst` поправьте, если ставили с `-Target`):

```powershell
$dst = "$env:LOCALAPPDATA\rus-transcriber\cache\gigaam"
New-Item -ItemType Directory -Force -Path $dst | Out-Null
$ProgressPreference = 'SilentlyContinue'
$files = [ordered]@{
  "emo.ckpt"                    = "7ce76f9535cb254488985057c0d33006"
  "v3_e2e_rnnt.ckpt"            = "2730de7545ac43ad256485a462b0a27a"
  "v3_e2e_rnnt_tokenizer.model" = "3b3bf8370e882885d79731592fc99f98"
}
foreach ($name in $files.Keys) {
  $final = Join-Path $dst $name
  if (Test-Path $final) {
    $existing = (Get-FileHash $final -Algorithm MD5).Hash.ToLower()
    if ($existing -eq $files[$name]) { Write-Host "уже проверен: $name"; continue }
    throw "Файл уже существует, но его md5 не совпал; не заменяю автоматически: $final"
  }
  $part = Join-Path $dst "$name.part"
  curl.exe -L --fail --output $part "http://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/$name"
  $md5 = (Get-FileHash $part -Algorithm MD5).Hash.ToLower()
  if ($md5 -ne $files[$name]) { Remove-Item $part -Force; throw "md5 не совпал: $name" }
  Move-Item $part $final
  Write-Host "OK $name"
}
```

`curl.exe` входит в Windows 10/11. Если его нет, замените строку загрузки на
`Invoke-WebRequest -Uri "http://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/$name" -OutFile $part`.

Суммы взяты из словаря `_MODEL_HASHES` пакета `gigaam`
(`.venv\Lib\site-packages\gigaam\__init__.py`). Если пакет обновился, берите
актуальные значения оттуда. Для второй модели в режиме PyTorch понадобится
ещё `v3_e2e_ctc.ckpt` (сумма `367074d6498f426d960b25f49531cf68`) и её
токенизатор.

## 3. Проверить загрузку

```powershell
Get-FileHash "$dst\emo.ckpt" -Algorithm MD5
Get-FileHash "$dst\v3_e2e_rnnt.ckpt" -Algorithm MD5
```

Суммы должны совпасть с таблицей выше. Затем проверьте, что модели
загружаются (пакет ещё раз сверит md5 и выдаст `Model checksum failed`,
если файл повреждён):

```powershell
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); import paths, gigaam; gigaam.load_model('emo', download_root=str(paths.GIGAAM_DIR)); print('emo OK')"
.\.venv\Scripts\python.exe -c "import sys; sys.path.insert(0, '.'); import paths, gigaam; gigaam.load_model('v3_e2e_rnnt', download_root=str(paths.GIGAAM_DIR)); print('rnnt OK')"
```

После этого эмоции и PyTorch-эталон в тесте скорости работают без скачивания
из сети. Повторять загрузку не нужно: файлы переиспользуются.

## 4. Когда сертификат продлят

Ничего удалять не нужно: загрузчик увидит готовые файлы и пропустит
скачивание. Проверять сертификат можно командой из раздела 1. Если сертификат
снова стал действительным, остальные модели скачаются обычным путём по HTTPS.

## 5. Если ошибка не про срок, а про недоверенный корень

Сообщение `unable to get local issuer certificate` (код 20) означает, что
сертификат сервера живой, но клиент не знает его корень. Так бывает у сайтов,
подписанных НУЦ Минцифры: в хранилище Mozilla (а значит, и в `certifi`) этого
корня нет.

Правильное лечение — установить корневой сертификат в хранилище Windows:

```
https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt
```

При импорте `paths.py` проект пытается подключить `truststore` и системные
сертификаты Windows (`truststore.inject_into_ssl()`). Если импорт и подключение
прошли успешно, Python будет проверять цепочку с этим хранилищем без правки
пакетов. Если `truststore` отсутствует или подключение завершилось ошибкой,
этот механизм не включится. Не добавляйте сертификат в файл
`certifi` через `>>`: при обновлении пакета правка пропадёт, а `pip` использует
свой отдельный bundle (`.venv\Lib\site-packages\pip\_vendor\certifi\cacert.pem`).

## 6. Если ошибка при установке пакетов (pip)

Если установщик падает на скачивании PyTorch или других пакетов с
`pypi.org` / `download.pytorch.org`:

1. Проверьте версию: ` .\.venv\Scripts\python.exe -c "import certifi; print(certifi.__version__)"`.
   Она должна быть актуальной; обновление — отдельное действие.
2. Посмотрите настройки: ` .\.venv\Scripts\python.exe -m pip config list -v` —
   нет ли своего `cert=` или старого bundle.
3. Проверьте, не перехватывает ли TLS прокси или антивирус: у нормальных
   сайтов цепочка должна быть от GlobalSign / Amazon / Let's Encrypt, а не от
   локального антивируса.

Команды ручной загрузки из раздела 2 предназначены **только для весов GigaAM**
с CDN Sber. Они не устанавливают `torch`, `pip` или зависимости из PyPI и не
исправляют сертификат `download.pytorch.org`. Для кода 10 на PyPI/PyTorch
сверьте системные часы и проверьте сертификат хоста; если просрочен именно
серверный сертификат, обновление certifi это не исправит. Если ошибка возникает
на многих независимых сайтах, проверьте прокси/антивирус. Для кода 20 определите
издателя цепочки и установите доверенный корень только если это действительно
нужный CA. Проверка TLS остаётся включённой.

Подсказку с командами всегда можно получить так:

```powershell
.\.venv\Scripts\python.exe download_hint.py
```

## Ссылки

- Пример публичного отчёта о просроченном сертификате CDN:
  <https://github.com/salute-developers/GigaAM/issues/99>
- Корневой сертификат НУЦ Минцифры:
  <https://gu-st.ru/content/lending/russian_trusted_root_ca_pem.crt>
