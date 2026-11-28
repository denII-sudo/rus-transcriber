"""БД-1 (S-2, О-7): восстановление из резерва поверх базы с непустым -wal, при открытом писателе и при сбое.

Сценарии плана sqlite-audit: чистая база; обрыв писателя после подтверждённых транзакций (-wal не пуст, дочерний
процесс завершается os._exit); то же после роста базы; писатель держит соединение; сбой между переименованием
прежней базы и публикацией. Проверки: «before-restore» содержит всё подтверждённое; восстановленная база — ровно
резерв с новыми корнями и установкой, integrity_check=ok; рядом с ней нет чужих -wal/-shm; при отказе прежняя база
и её служебные файлы побайтно не изменились. Только синтетика во временной папке. Поведение переименования
открытого файла — Windows (CI на Windows); вне Windows замена базы — отказ, его проверяет отдельный тест.
"""
import atexit
import hashlib
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

_TMP = tempfile.mkdtemp(prefix="rt-restore-wal-")
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)
if "paths" not in sys.modules:                       # данные программы — во временную папку до любого импорта
    os.environ["RUS_TRANSCRIBER_DATA"] = _TMP
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import core  # noqa: E402
import core_conflicts  # noqa: E402
import storage  # noqa: E402

PAD = "x" * 64 * 1024          # строка пользователя занимает много страниц: -wal и рост базы заметны
WRITER = """
import os, sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import storage
db, n, how = sys.argv[2], int(sys.argv[3]), sys.argv[4]
store = storage.open_store(db)
for i in range(n):
    store.create_user(f"writer-{os.getpid()}-{i}" + "x" * 64 * 1024)
print("committed", n, flush=True)
if how == "close":
    store.close()
elif how == "kill":
    os._exit(0)                  # без закрытия: подтверждённые кадры остаются в -wal
else:                            # hold: держать соединение, пока тест не закроет stdin
    sys.stdin.readline()
    store.close()
"""
WINDOWS = os.name == "nt"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest() if os.path.lexists(path) else None


def side_files(db):
    return {suffix: os.path.lexists(str(db) + suffix) for suffix in ("-wal", "-shm")}


class RestoreCase(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="rt-restore-wal-case-"))
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        for name in ("r1", "r2", "inspect"):
            (self.base / name).mkdir()
        self.db = self.base / "state.sqlite3"
        store = storage.initialize(self.db, initial_display_name="Тест")
        try:
            store.register_root(self.base / "r1")
            self.user_id = store.current_identity()["user_id"]
            self.backup = self.base / "b1.sqlite3"
            store.backup(self.backup, allow_damaged_copies=True)
            self.root_id = store.db.execute("SELECT root_id FROM allowed_roots").fetchone()[0]
        finally:
            store.close()
        self.inspected = 0

    def writer(self, n, how):
        """Отдельный процесс пишет n пользователей и закрывается, обрывается или держит соединение."""
        args = [sys.executable, "-I", "-B", "-c", WRITER, str(REPO), str(self.db), str(n), how]
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
        if how == "hold":
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, env=env)

            def release():
                if proc.poll() is None:
                    proc.stdin.close()
                    proc.wait(timeout=60)
                proc.stdout.close()
            self.addCleanup(release)
            self.assertEqual(proc.stdout.readline().strip(), f"committed {n}")
            return release
        result = subprocess.run(args, capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), f"committed {n}")
        return None

    def counted(self, db):
        """Пользователей и integrity_check в копии набора файлов базы (оригиналы не открываются)."""
        self.inspected += 1
        folder = self.base / "inspect" / str(self.inspected)
        folder.mkdir()
        for suffix in ("", "-wal", "-shm"):
            if os.path.lexists(str(db) + suffix):
                shutil.copy2(str(db) + suffix, folder / ("copy.sqlite3" + suffix))
        con = sqlite3.connect(folder / "copy.sqlite3")
        try:
            return (con.execute("SELECT count(*) FROM local_users").fetchone()[0],
                    con.execute("PRAGMA integrity_check").fetchone()[0])
        finally:
            con.close()

    def restore(self):
        return core.restore_from_backup(self.backup, self.db, {self.root_id: str(self.base / "r2")}, self.user_id,
                                        replace=True)

    def names(self):
        return sorted(path.name for path in self.base.iterdir() if path.name.startswith("state.sqlite3"))

    def check_restored(self, result):
        """Восстановленная база — ровно резерв: 1 пользователь, новый корень, новая установка, без чужого -wal."""
        self.assertEqual(side_files(self.db), {"-wal": False, "-shm": False})
        self.assertEqual(self.counted(self.db), (1, "ok"))
        store = storage.open_store(self.db)
        try:
            self.assertEqual(store.current_identity()["installation_id"], result["installation_id"])
            self.assertEqual([row[0] for row in store.db.execute("SELECT path FROM allowed_roots")],
                             [os.path.normcase(str((self.base / "r2").resolve()))])
            self.assertEqual(store.db.execute("PRAGMA integrity_check").fetchone()[0], "ok")
        finally:
            store.close()
        self.assertFalse([name for name in self.names() if ".incomplete-" in name])

    def check_previous(self, result, users):
        """«before-restore» — прежняя база целиком, со всеми подтверждёнными транзакциями, без служебных файлов."""
        saved = Path(result["previous_saved_as"])
        self.assertEqual(saved.parent, self.db.parent)
        self.assertTrue(saved.name.startswith("state.sqlite3.before-restore-"))
        self.assertEqual(side_files(saved), {"-wal": False, "-shm": False})
        self.assertEqual(self.counted(saved), (users, "ok"))
        return saved


@unittest.skipUnless(WINDOWS, "замена базы при восстановлении — только Windows")
class RestoreOverWalTests(RestoreCase):
    def test_clean_database(self):
        before = digest(self.db)
        result = self.restore()
        saved = self.check_previous(result, 1)
        self.assertEqual(digest(saved), before)          # переименованный файл, а не копия
        self.check_restored(result)
        self.assertNotIn("warning", result)

    def test_writer_killed_with_committed_wal(self):
        self.writer(3, "kill")
        self.assertTrue(os.path.getsize(str(self.db) + "-wal") > 0)
        self.assertEqual(self.counted(self.db), (4, "ok"))
        result = self.restore()
        self.check_previous(result, 4)                   # прежде: 1, три подтверждённые транзакции терялись
        self.check_restored(result)                      # прежде: 4 пользователя — чужой -wal наложился

    def test_grown_database_then_writer_killed(self):
        self.writer(40, "close")
        self.writer(3, "kill")
        self.assertEqual(self.counted(self.db), (44, "ok"))
        result = self.restore()
        self.check_previous(result, 44)                  # прежде: 41
        self.check_restored(result)                      # прежде: integrity_check «Page N: never used»

    def test_writer_holds_connection_nothing_changes(self):
        release = self.writer(3, "hold")
        files = {suffix: digest(str(self.db) + suffix) for suffix in ("", "-wal", "-shm")}
        self.assertIsNotNone(files["-wal"])
        names = self.names()
        with self.assertRaises(core.CoreError) as caught:
            self.restore()
        self.assertIn("другая программа", str(caught.exception))
        self.assertEqual({suffix: digest(str(self.db) + suffix) for suffix in ("", "-wal", "-shm")}, files)
        self.assertEqual(self.names(), names)            # ни before-restore, ни .incomplete-* (прежде — оба)
        self.assertEqual(list((self.base / "r2").iterdir()), [])
        release()
        self.assertEqual(self.counted(self.db), (4, "ok"))

    def test_failure_between_rename_and_publish_returns_the_set(self):
        before = digest(self.db)
        with mock.patch.object(core_conflicts, "_publish", side_effect=OSError("synthetic failure")):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        self.assertIn("прежняя база на месте", str(caught.exception))
        self.assertEqual(digest(self.db), before)
        names = self.names()
        self.assertFalse([name for name in names if ".before-restore-" in name])
        staged = [name for name in names if ".incomplete-" in name]
        self.assertEqual(len(staged), 1)                 # подготовленный файл остаётся, путь в сообщении
        self.assertIn(staged[0], str(caught.exception))
        self.assertEqual(self.counted(self.db), (1, "ok"))

    def test_failure_after_crash_keeps_every_committed_transaction(self):
        self.writer(3, "kill")
        with mock.patch.object(core_conflicts, "_publish", side_effect=OSError("synthetic failure")):
            with self.assertRaises(core.CoreError):
                self.restore()
        self.assertFalse([name for name in self.names() if ".before-restore-" in name])
        self.assertEqual(side_files(self.db), {"-wal": False, "-shm": False})   # журнал внесён в базу
        self.assertEqual(self.counted(self.db), (4, "ok"))

    def test_side_files_follow_an_unreadable_database(self):
        """Прежний файл не база (испорчен или пуст): набор откладывается как есть, вместе со служебными файлами;
        SQLite его не открывает (при открытии он удалил бы лежащий рядом -wal)."""
        for content in (b"not a database" * 100, b""):
            with self.subTest(size=len(content)):
                self.db.unlink()
                self.db.write_bytes(content)
                Path(str(self.db) + "-wal").write_bytes(b"stray journal")
                result = self.restore()
                saved = Path(result["previous_saved_as"])
                self.assertEqual(saved.read_bytes(), content)
                self.assertEqual(Path(str(saved) + "-wal").read_bytes(), b"stray journal")
                self.assertTrue(any("отложена как есть" in line for line in result["warning"]))
                self.check_restored(result)


@unittest.skipUnless(WINDOWS, "замена базы при восстановлении — только Windows")
class RefusalAfterCopyTests(RestoreCase):
    """Р258-2, Р258-5: отказы после подготовки копии называют, что осталось; возврат набора при сбое."""

    def files(self):
        return {suffix: digest(str(self.db) + suffix) for suffix in ("", "-wal")}

    def assert_names_staged(self, message):
        staged = [name for name in self.names() if ".incomplete-" in name]
        self.assertEqual(len(staged), 1)
        self.assertIn(staged[0], message)
        self.assertIn("Осталось:", message)
        self.assertNotIn("Ничего не изменено", message)

    def test_writer_after_probe_is_refused_before_documents(self):
        release = self.writer(3, "hold")
        files = self.files()
        with mock.patch.object(storage, "file_held_elsewhere", return_value=False):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        message = str(caught.exception)
        self.assertIn("другая программа", message)
        self.assert_names_staged(message)
        self.assertIn("-shm", message)
        self.assertNotIn("документы", message)
        self.assertEqual(list((self.base / "r2").iterdir()), [])
        self.assertEqual(self.files(), files)
        self.assertFalse([name for name in self.names() if ".before-restore-" in name])
        release()
        self.assertEqual(self.counted(self.db), (4, "ok"))

    def test_writer_after_step_2_is_refused_by_the_hold(self):
        release = self.writer(3, "hold")
        files = self.files()
        with mock.patch.object(storage, "file_held_elsewhere", return_value=False), \
                mock.patch.object(core_conflicts, "_settle_previous", return_value=None):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        message = str(caught.exception)
        self.assertIn("только что открыла другая программа", message)
        self.assert_names_staged(message)
        self.assertEqual(list((self.base / "r2").iterdir()), [])
        self.assertEqual(self.files(), files)
        release()
        self.assertEqual(self.counted(self.db), (4, "ok"))

    def test_previous_base_is_held_while_documents_are_written(self):
        """Окно между проверкой и переименованием закрыто: пока пишутся документы, прежнюю базу не открыть."""
        import documents
        tried = []
        original = documents.materialize_pending

        def materialize(*args, **kwargs):
            try:
                con = sqlite3.connect(self.db, timeout=0)
                try:
                    con.execute("SELECT count(*) FROM local_users").fetchone()
                    tried.append("opened")
                finally:
                    con.close()
            except sqlite3.Error as exc:
                tried.append(type(exc).__name__)
            return original(*args, **kwargs)
        with mock.patch.object(documents, "materialize_pending", side_effect=materialize):
            result = self.restore()
        self.assertEqual(tried, ["OperationalError"])
        self.check_previous(result, 1)
        self.check_restored(result)

    def test_failed_side_file_rename_returns_the_set(self):
        self.db.unlink()
        self.db.write_bytes(b"not a database")
        Path(str(self.db) + "-wal").write_bytes(b"stray journal")
        original = os.rename

        def rename(source, target):
            if str(source).endswith("-wal") and ".before-restore-" in str(target):
                raise PermissionError("synthetic")
            original(source, target)
        with mock.patch.object(core_conflicts, "_rename", side_effect=rename):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        self.assertIn("прежняя база на месте", str(caught.exception))
        self.assert_names_staged(str(caught.exception))
        self.assertEqual(self.db.read_bytes(), b"not a database")
        self.assertEqual(Path(str(self.db) + "-wal").read_bytes(), b"stray journal")
        self.assertFalse([name for name in self.names() if ".before-restore-" in name])

    def test_failed_publish_and_failed_return_name_both_files(self):
        original = os.rename

        def rename(source, target):
            if Path(target) == self.db:
                raise PermissionError("synthetic")
            original(source, target)
        with mock.patch.object(core_conflicts, "_publish", side_effect=OSError("synthetic")), \
                mock.patch.object(core_conflicts, "_rename", side_effect=rename):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        message = str(caught.exception)
        saved = [name for name in self.names() if ".before-restore-" in name]
        self.assertEqual(len(saved), 1)
        self.assertIn(saved[0], message)
        self.assertIn("должен называться", message)
        self.assert_names_staged(message)
        self.assertFalse(self.db.exists())
        self.assertEqual(self.counted(self.base / saved[0]), (1, "ok"))      # ничего не удалено

    def test_foreign_empty_database_at_the_name_says_what_to_rename(self):
        """Р258-10: сторонняя программа SQLite («создать, если нет») между переименованием и публикацией создаёт
        пустую базу на месте рабочей. Отказ говорит, какой файл во что переименовать; по его словам база
        возвращается. Окно кодом не закрывается."""
        self.check_foreign_database(foreign_wal=False)

    def test_foreign_database_with_its_journal_says_if_they_exist(self):
        """Н258-1: чужие -wal/-shm SQLite убирает, когда чужая программа закрывается, — шаг «если они есть»."""
        self.check_foreign_database(foreign_wal=True)

    def check_foreign_database(self, foreign_wal):
        original = core_conflicts._move_aside

        def move_aside(destination):
            moved = original(destination)
            sqlite3.connect(destination).close()     # пустой файл на месте базы
            if foreign_wal:
                Path(str(destination) + "-wal").write_bytes(b"foreign journal")
            return moved
        with mock.patch.object(core_conflicts, "_move_aside", side_effect=move_aside):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        message = str(caught.exception)
        self.assertIn("появился чужой файл", message)
        self.assertEqual("если они есть" in message, foreign_wal)
        self.assert_names_staged(message)
        self.assertEqual(self.db.stat().st_size, 0)
        saved = [name for name in self.names() if ".before-restore-" in name]
        self.assertEqual(len(saved), 1)
        quoted = re.findall(r"«([^»]+)»", message.split("Осталось:")[0])
        foreign = str(self.db) + ".foreign-"
        steps = list(zip(quoted[1::2], quoted[2::2]))
        self.assertEqual(quoted[0], str(self.db))
        self.assertEqual(steps[0][0], str(self.db))
        self.assertTrue(steps[0][1].startswith(foreign))
        if foreign_wal:
            self.assertEqual(steps[1], (str(self.db) + "-wal", steps[0][1] + "-wal"))
        self.assertEqual(steps[-1], (str(self.base / saved[0]), str(self.db)))
        self.assertEqual(len(steps), 3 if foreign_wal else 2)
        for source, target in steps:                 # человек делает, что сказано
            os.rename(source, target)
        self.assertFalse(Path(str(self.db) + "-wal").exists())
        self.assertEqual(self.counted(self.db), (1, "ok"))
        self.assertEqual(Path(steps[0][1]).stat().st_size, 0)

    def test_second_name_left_after_publish_is_a_strong_warning(self):
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            if ".incomplete-" in path.name:
                raise PermissionError("synthetic")
            return original(path, *args, **kwargs)
        with mock.patch.object(Path, "unlink", autospec=True, side_effect=unlink):
            result = self.restore()
        staged = [self.base / name for name in self.names() if ".incomplete-" in name]
        self.assertEqual(len(staged), 1)
        self.assertTrue(os.path.samefile(staged[0], self.db))
        text = " ".join(result["warning"])
        self.assertIn("не открывайте его ничем, пока работает программа", text)
        self.assertIn("Удалите его после закрытия программы", text)

    def test_read_only_previous_base_gets_the_right_reason(self):
        self.writer(3, "kill")
        os.chmod(self.db, stat.S_IREAD)
        self.addCleanup(os.chmod, self.db, stat.S_IREAD | stat.S_IWRITE)
        files = {suffix: digest(str(self.db) + suffix) for suffix in ("", "-wal", "-shm")}
        names = self.names()
        with self.assertRaises(core.CoreError) as caught:
            self.restore()
        self.assertIn("защищён от записи", str(caught.exception))
        self.assertIn("Ничего не изменено", str(caught.exception))
        self.assertEqual({suffix: digest(str(self.db) + suffix) for suffix in ("", "-wal", "-shm")}, files)
        self.assertEqual(self.names(), names)

    def test_damaged_backup_is_refused_before_any_document(self):
        import documents
        with mock.patch.object(storage, "validate_copy", side_effect=storage.StorageError("integrity check failed")), \
                mock.patch.object(documents, "materialize_pending") as materialize:
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        materialize.assert_not_called()
        self.assert_names_staged(str(caught.exception))
        self.assertNotIn("документы", str(caught.exception))
        self.assertFalse([name for name in self.names() if ".before-restore-" in name])


class RestoreRefusalTests(RestoreCase):
    def test_backup_with_its_own_journal_is_refused_untouched(self):
        """Р258-6: чей журнал рядом с резервом, не узнать — отказ до любых изменений."""
        Path(str(self.backup) + "-wal").write_bytes(b"journal of unknown origin")
        names = self.names()
        before = digest(self.db)
        with self.assertRaises(core.CoreError) as caught:
            self.restore()
        self.assertIn("журнал", str(caught.exception))
        self.assertIn("Ничего не изменено", str(caught.exception))
        self.assertEqual((self.names(), digest(self.db)), (names, before))

    def test_orphan_wal_without_database_is_refused_untouched(self):
        self.db.rename(self.base / "elsewhere.sqlite3")
        Path(str(self.db) + "-wal").write_bytes(b"orphan")
        names = self.names()
        with self.assertRaises(core.CoreError) as caught:
            self.restore()
        self.assertIn("без самой базы", str(caught.exception))
        self.assertEqual(self.names(), names)
        self.assertEqual(Path(str(self.db) + "-wal").read_bytes(), b"orphan")

    def test_replacing_outside_windows_is_refused_untouched(self):
        before = digest(self.db)
        names = self.names()
        with mock.patch.object(core_conflicts, "_REPLACE_SUPPORTED", False):
            with self.assertRaises(core.CoreError) as caught:
                self.restore()
        self.assertIn("только в Windows", str(caught.exception))
        self.assertEqual((digest(self.db), self.names()), (before, names))

    def test_new_destination_needs_no_replace(self):
        result = core.restore_from_backup(self.backup, self.base / "fresh.sqlite3",
                                          {self.root_id: str(self.base / "r2")}, self.user_id)
        self.assertIsNone(result["previous_saved_as"])
        self.assertEqual(self.counted(self.base / "fresh.sqlite3"), (1, "ok"))
        self.assertEqual(side_files(self.base / "fresh.sqlite3"), {"-wal": False, "-shm": False})


if __name__ == "__main__":
    unittest.main()
