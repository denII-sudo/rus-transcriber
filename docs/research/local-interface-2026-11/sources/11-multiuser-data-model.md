# Источник 11. Многопользовательская структура данных

Отчёт поискового субагента (модель Sonnet), 22.11.2026. Приведён как получен; не перепроверен,
кроме оговорённого в [README.md](../README.md). Обращения «вы», «твой» — к заказчику поиска.

## Исследование: что заложить в схему SQLite для многопользовательности

Только чтение, файлы не создавались. Пометки: [Д] первоисточник (документация, исходники), [Ф] блоги и форумы, [?] мой вывод. Сайты с текстами 152-ФЗ и 572-ФЗ не открылись (сбой сокета). Юридическая часть опирается на выдачу поиска, это общие ориентиры, а не консультация.

### 1. Что дорого добавлять потом, что дёшево

**Дорого (теряется информация или переписывается всё):**
- **Автор действия.** Кто ответил на карточку, кто правил текст. Прошлое не восстановить, а старые строки придётся помечать «владелец» [?].
- **Идентификатор пространства (арендатора) в каждой таблице.** Он входит в составные уникальные ключи, внешние ключи и индексы. Для PostgreSQL row-level security он нужен прямо в строке: [Д] документация советует политики, зависящие только от текущей строки, это проще и быстрее (PostgreSQL docs, ddl-rowsecurity).
- **Идентификаторы.** Целые автоинкременты сталкиваются при слиянии баз и импорте. У вас уже есть export_group/import_group между установками [?]. UUIDv7 сортируется по времени и хорош для индексов [Д RFC 9562, май 2024]. Он раскрывает время создания и не годится как секрет [Д]. Immich использует UUIDv7 для таблиц-журналов [Д].
- **Ответ как отдельная строка, а не поле карточки.** Иначе несколько людей на одной карточке не уместить. Argilla хранит несколько ответов на запись со статусами draft/submitted/discarded [Д]. Label Studio ставит `completed_by` на каждую аннотацию [Д].
- **Согласие субъекта голоса**, отдельно от учётной записи (см. п. 6).

**Дёшево позже:** роли и права на объект, таблица захвата карточек (lease), журнал аудита как отдельная таблица, мягкое удаление, квоты.

Open WebUI сначала держал права JSON-колонкой в каждой ресурсной таблице, затем заменил её нормализованной таблицей `access_grant` [Д docs.openwebui.com, схема до v0.10.0]. Вывод: не кладите права JSON-полем, а делайте таблицу прав, когда они понадобятся.

### 2. Модели разделения

- **Общая база со столбцом `workspace_id`.** 37signals начал с SQLite на аккаунт для Fizzy. Из-за маршрутизации и репликации (одна копия принимает запись) и нехватки времени он запустился на MySQL с `account_id` и скриптами переноса из SQLite [Д dev.37signals.com, даты на странице нет]. Tenancy-код они вынесли в gem Active Record Tenanted. Из ONCE я подтвердил только то, что Campfire работает на SQLite.
- **Файл на арендатора.** Turso (Per User Starter, групповые токены, запрос присоединяет до 10 баз, часть страниц помечена deprecated) [Д]. Cloudflare D1: база на пользователя или арендатора, каждая на одном Durable Object, рост за 2026 год [Д, 8 окт. 2026]. Сам SQLite называет «файл на пользователя» способом масштабирования [Д whentouse]. Плюсы: изоляция и удаление арендатора удалением файла. Минусы: миграции ×N, запросы между арендаторами, маршрутизация.
- **Схема на арендатора в PostgreSQL + RLS.** Таблица-владелец обходит RLS без `FORCE`, роли `BYPASSRLS` и суперпользователь тоже. Проверки уникальных и внешних ключей RLS обходят и могут утекать. `SET LOCAL app.tenant_id` в транзакции безопаснее с пулом соединений [Д].
- **Litestream** (поток резервных копий в объектное хранилище, актуален, v0.5.x в октябре 2025) [Д/Ф]. **LiteFS Cloud закрыт 15.10.2024**, статус самого LiteFS неясен [Ф community.fly.io]. Это не опора для репликации.
- **Для вас [?]:** на этапе 1 одна база с `workspace_id`. Файловая раскладка по `workspace_id` оставит открытым путь «файл на группу».
- **Дешёвый путь в PostgreSQL [?]:**
  - `STRICT`-таблицы (SQLite ≥ 3.37.0, 27.11.2021) [Д];
  - явные типы, булевы как 0/1 с CHECK;
  - время как ISO-8601 UTC в TEXT или целое, а не «как придётся»;
  - JSON только валидный;
  - `foreign_keys=ON`;
  - `PRAGMA user_version`;
  - не опираться на rowid и AUTOINCREMENT.
- pgloader сам приводит типы и сбрасывает sequences, но разбор текстовых и целых дат нужно проверять на реальных значениях [Д pgloader docs].

### 3. Опыт открытых программ

- **Immich** [Д, schema/tables]. Есть `user` (`isAdmin`, `deletedAt`, `quotaSizeInBytes`, `updateId`) и `session` (`token` bytea, `deviceType`, `deviceOS`, `appVersion`, `parentId`, `expiresAt`). Есть `api_key`, `passkey`, `auth_challenge`, `partner`, `shared_link`, `person_user`. Есть и «аудит-надгробия»: `user_audit`, `album_audit` с `id`, `userId`, `deletedAt`. Они нужны клиентам, чтобы синхронизировать удаления. `sync_checkpoint` хранит точки синхронизации.
- **Jellyfin** [Д]. Таблица `Device` содержит `UserId`, `AccessToken`, `DeviceId`, `AppName`, `IsActive`, `DateLastActivity`. Рядом `ApiKeys` и `DeviceOptions`. Всё в одной `jellyfin.db` только с 10.11 (EF Core), миграция необратима, перед ней нужна копия [Ф jellywatch, официальный релиз].
- **Home Assistant** [Д]. Токены доступа короткие. Refresh-токен живёт 90 дней без использования. Отзыв инвалидирует и порождённые токены. Долгоживущий токен действует до 10 лет, а HA не хранит его значение. Провайдер `trusted_networks` даёт вход без пароля и без MFA [Д].
- **Paperless-ngx** [Д docs.paperless-ngx.com/usage]. Права глобальные плюс на объект: владелец, просмотр и правка для пользователей и групп. Документы из папки приёма по умолчанию без владельца, задавать его можно через workflows.
- **Memos** [Д, LATEST.sql]. В `memo` есть `creator_id` и `visibility`. Появились `space`, `space_member`, `memo_share` (`expires_ts`), `user_identity`.
- **Label Studio** [Д, исходники]. `Annotation.completed_by` и `updated_by`. `Task.overlap`. Таблица `TaskLock` (`user`, `task`, `expire_at`, TTL из настроек).
- **Actual Budget** [Ф/Д jlongster.com]. Журнал `messages_crdt`, часы HLC, деревья Меркла, сервер как «ретранслятор».
- **Linkding, Audiobookshelf, Nextcloud** (`oc_share`, `oc_authtoken`, `oc_storages` [?]): первоисточники не открылись, выводов по ним нет.
- **Общий урок [?]:** позднее добавление идёт через «владелец = NULL значит всем» и через догоняющие миграции, и именно это ломает приватность. Начинайте с `owner/workspace NOT NULL`.

### 4. Конкурентная правка

- Для карточек (дискретные решения) CRDT не нужен [?]. Достаточно ответа на пользователя и «принятого» решения, которое выбирает человек. Это совпадает с приоритетом «человек важнее» и правилом «не выбирать автора спорной реплики за человека».
- Для правки текста стенограммы хватает оптимистичной версии: запись с `base_revision` отклоняется при расхождении, а у вас уже есть конфликты документов.
- CRDT окупается при одновременной правке одного абзаца и работе офлайн [?]. Yjs заточен под совместное редактирование текста [Д yjs.dev], Automerge под офлайн-слияние документов [Д automerge.org]. Стоимость: рост истории и непрозрачные для человека слияния [?].
- **Захват карточек.** Label Studio блокирует задачу на время работы человека (TTL-блокировка, `expire_at`), а предупреждает, что две вкладки обходят `overlap` [Д]. Argilla оставляет запись в очереди, пока нет `min_submitted` ответов, и признаёт возможность лишних ответов при одновременной отправке [Д]. Prodigy по умолчанию отдаёт пример одному аннотатору, режимы перекрытия: `feed_overlap` и `annotations_per_task` [Д]. Разбор расхождений есть в Label Studio Enterprise (agreement, платно) [Д].

### 5. Аутентификация, этап 1

- **Сопряжение устройства.** Jellyfin Quick Connect: на новом устройстве 6-значный код, подтверждение с уже вошедшего устройства [Д]. Syncthing: ID устройства равен SHA-256 сертификата и должен быть добавлен с обеих сторон; локальное обнаружение не защищено [Д].
- **Токены.** Хранить хеш токена, срок жизни, `last_seen`, `revoked_at`, имя устройства, пользователя [?] (по образцу `session` в Immich и `Device` в Jellyfin).
- **Passkeys/WebAuthn.** Нужен безопасный контекст: HTTPS или localhost. RP ID должен быть доменом, **IP-адрес не подходит**. `.local`, по отчётам, вызывает SecurityError [Ф web.dev, Ory, W3C-список]. Для домашней сети нужно настоящее доменное имя (split-DNS) и доверенный сертификат; самоподписанный не проверен [?]. Записи ключа по W3C WebAuthn-3: credential id, public key, signCount, transports, backup-флаги, user handle [Д].
- Вход «из доверенной сети без пароля» для биометрии не годится (см. HA) [?].

### 6. Чувствительные данные

- **GDPR ст. 9(1):** биометрия для однозначной идентификации человека относится к особой категории. Основание для обработки, как правило, явное согласие 9(2)(a) [Д gdpr-info.eu]. Ст. 2(2)(c) исключает «чисто личную или домашнюю деятельность» [Д]. Этап 1 близок к ней, этап 2 как сервис уже нет [?].
- **152-ФЗ.** Ст. 11: биометрия обрабатывается по письменному согласию, исключения перечислены в ч. 2 [Ф ppt.ru, kontur.ru]. Отказ от биометрии не должен лишать услуги [Ф]. Ст. 18 ч. 5: первичная запись ПДн граждан РФ на серверах в РФ, штрафы крупные [Ф, не проверено по официальному тексту]. 572-ФЗ (ЕБС): источники расходятся в номере статьи и охвате [?].
- **Следствия для архитектуры [?]:**
  - согласие привязано к *субъекту голоса*, а не к учётной записи;
  - отзыв согласия ведёт к физическому удалению профилей и производных, поэтому «мягкого удаления» для биометрии нет;
  - у резервных копий должна быть политика удаления;
  - каталоги биометрии раздельные по `workspace_id`;
  - возможность ключа шифрования на пространство;
  - журнал доступа к профилям;
  - профили не уходят в облачные модели;
  - привязка пространства к региону хранения.

### 7. SQLite при нескольких пользователях

- WAL: один писатель, читатели не блокируют писателя, **только на одном хосте, не по сетевой ФС** [Д sqlite.org/wal]. Много перекрывающихся читателей мешают чекпоинту, WAL растёт [Д].
- Использовать `BEGIN IMMEDIATE` для записи, `busy_timeout`, короткие транзакции. Повышение чтения до записи падает с `SQLITE_BUSY`, если снимок устарел, и тайм-аут тут не помогает [Д lang_transaction].
- `synchronous=NORMAL` в WAL нормален, `foreign_keys` выключены по умолчанию [Д pragma].
- Запас: сайт до ~100 тыс. запросов в сутки без проблем [Д whentouse]. Для 2–3 человек этого хватает. Не держите транзакции открытыми во время распознавания или вызова модели [?].

### 8. Итог

**Заложить сейчас:**
1. `workspace(id, name, created_at)`. Это ваша «группа». Столбец `workspace_id NOT NULL` во всех таблицах. Зачем: арендатор, раскладка файлов, RLS.
2. `user(id, display_name, status, created_at, disabled_at)` и `membership(user_id, workspace_id, role)`. Зачем: автор, роль владельца и доверенного.
3. `subject` (человек, чей голос) отдельно от `user` + `consent(subject_id, scope, given_at, withdrawn_at, basis)`. Зачем: биометрия неучастников.
4. `device(id, user_id, token_hash, name, created_at, last_seen_at, expires_at, revoked_at)`.
5. Идентификаторы: приложение генерирует UUIDv7/ULID для сущностей. Время: UTC. `STRICT`.
6. Автор: `created_by`, `actor_kind` (человек, модель, система), `device_id` у ревизий, ответов, правок. Для существующих строк это «владелец», backfill один раз.
7. `answer(id, card_id, user_id, status, value, version, created_at)` плюс указатель «принятый ответ».
8. Журнал событий: добавить `actor` и `workspace`, только добавление записей, без текста содержимого.
9. `version` на изменяемых строках и `base_revision` при записи.
10. Мягкое удаление `deleted_at`/`deleted_by` только для небиометрических сущностей.

**Отложить:** RLS и PostgreSQL, таблица прав на объект, захват карточек (новая таблица), CRDT/HLC, таблица passkey, OIDC, квоты, шифрование на арендатора, база на арендатора.

**Вопросы владельцу:**
- Субъекты голоса бывают среди тех 2–3 людей или это третьи лица? От кого есть согласие на хранение голоса?
- Может ли доверенный видеть чужие записи и профили или каждый только свои группы?
- Кто принимает итоговое решение по спорной карточке, один владелец или любой?
- Нужен ли доступ извне дома? Есть ли свой домен для HTTPS и passkeys?
- Должны ли голоса уходить за пределы РФ или в облачные модели?
- Что делать с резервными копиями при отзыве согласия?
- Этап 2 реалистичен в течение года? Это определяет, нужен ли `workspace_id` везде сейчас.

### Источники
- SQLite: https://www.sqlite.org/wal.html, https://www.sqlite.org/lang_transaction.html, https://www.sqlite.org/whentouse.html, https://www.sqlite.org/pragma.html, https://www.sqlite.org/stricttables.html
- PostgreSQL RLS: https://www.postgresql.org/docs/current/ddl-rowsecurity.html
- RFC 9562: https://www.rfc-editor.org/rfc/rfc9562.html
- pgloader: https://pgloader.readthedocs.io/en/latest/ref/sqlite.html
- 37signals: https://dev.37signals.com/rails-multi-tenancy/ , https://dev.37signals.com/fizzy-infrastructure/
- Turso: https://turso.tech/multi-tenancy ; D1: https://blog.cloudflare.com/d1-control-plane/ ; LiteFS: https://community.fly.io/t/sunsetting-litefs-cloud/20829 ; Litestream: https://litestream.io/
- Immich: https://github.com/immich-app/immich/tree/main/server/src/schema/tables
- Jellyfin: https://jellyfin.org/docs/general/server/quick-connect/ , Device.cs в github.com/jellyfin/jellyfin
- Home Assistant: https://contributors.home-assistant.io/core/internals/auth-api/ , https://www.home-assistant.io/docs/authentication/providers/
- Paperless-ngx: https://docs.paperless-ngx.com/usage
- Open WebUI: https://docs.openwebui.com/reference/database-schema.md
- Memos: raw.githubusercontent.com/usememos/memos/main/store/migration/sqlite/LATEST.sql
- Label Studio: исходники label_studio/tasks/models.py; https://docs.humansignal.com/guide/labeling
- Argilla: https://docs.argilla.io/latest/how_to_guides/distribution/ ; Prodigy: https://prodi.gy/docs/task-routing
- Actual: https://jlongster.com/using-crdts-in-the-wild ; Yjs: https://yjs.dev/ ; Automerge: https://automerge.org/docs/hello/
- WebAuthn: https://www.w3.org/TR/webauthn-3/ , https://web.dev/articles/webauthn-rp-id ; Syncthing: https://docs.syncthing.net/dev/device-ids.html
- GDPR: https://gdpr-info.eu/art-9-gdpr/ , https://gdpr-info.eu/art-2-gdpr/
- 152-ФЗ и 572-ФЗ (вторичные): ppt.ru, kontur.ru, rulaws.ru, zakonrf.info/zakon-o-personalnyh-dannyh/18/
