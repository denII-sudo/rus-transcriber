"""БД-9а (S-21, О-7): остатки после обрыва только находятся и показываются — ничего не удаляется и не открывается.

Шаблоны строгие: `<имя>.incomplete-<32 hex>` (и его -wal/-shm/-journal) рядом с базой и в db-backups,
`.rt-<32 hex>.tmp` в корнях документов. Старше суток — в списке, моложе — только счётчиком. Второе имя рабочей базы
(жёсткая ссылка, Р258-7) помечается и показывается сразу. По ссылкам и junction обход не идёт. Синтетика во
временной папке.
"""
import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import threading
import time
import unittest
import uuid
import http.client
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import core
import leftovers
import paths
import recordings
import storage

OLD = time.time() - 3 * 86400


def touch(path, data=b"x", old=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    if old:
        os.utime(path, (OLD, OLD))
    return path


def tree_state(folder):
    """Имена, размеры, время и суммы всех файлов: проверка «ничего не изменено и не создано»."""
    state = {}
    for item in sorted(Path(folder).rglob("*")):
        if item.is_file() and not item.is_symlink():
            info = item.stat()
            state[str(item.relative_to(folder))] = (info.st_size, info.st_mtime_ns,
                                                    hashlib.sha256(item.read_bytes()).hexdigest())
    return state


def make_link_to_folder(target, link):
    if os.name == "nt":
        import _winapi
        _winapi.CreateJunction(str(target), str(link))
    else:
        os.symlink(target, link, target_is_directory=True)


class FindLeftovers(unittest.TestCase):
    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="leftovers-", dir=os.environ.get("TEMP")))
        self.addCleanup(shutil.rmtree, self.base, True)
        self.data = self.base / "data"
        self.db = self.data / "state.sqlite3"
        self.data.mkdir()
        storage.initialize(self.db, initial_display_name="Тест").close()
        self.root = self.base / "Записи" / "a.wav.rus-transcriber"
        self.root.mkdir(parents=True)
        self.hex = uuid.uuid4().hex

    def found(self, report):
        return {Path(item["path"]).name: item for item in report["items"]}

    def test_only_program_names_older_than_a_day(self):
        copy = touch(self.data / f"state.sqlite3.incomplete-{self.hex}", b"c" * 100)
        touch(self.data / f"state.sqlite3.incomplete-{self.hex}-wal", b"w" * 10)
        touch(self.data / "db-backups" / f"r.sqlite3.incomplete-{uuid.uuid4().hex}-journal")
        temp = touch(self.root / f".rt-{self.hex}.tmp", b"t" * 7)
        touch(self.root / "sub" / f".rt-{uuid.uuid4().hex}.tmp")
        touch(self.data / f"state.sqlite3.incomplete-{uuid.uuid4().hex}", old=False)     # свежий: в работе
        touch(self.root / f".rt-{uuid.uuid4().hex}.tmp", old=False)
        for name in (f"state.sqlite3.incomplete-{self.hex[:31]}", f"state.sqlite3.incomplete-{uuid.uuid4().hex.upper()}",
                     f"state.sqlite3.incomplete-{self.hex}.bak", "state.sqlite3.before-restore-20261128-100000",
                     f"x.incomplete-{self.hex}-wal2", "state.sqlite3-wal"):
            touch(self.data / name)
        for name in (f".rt-{self.hex[:31]}.tmp", f"rt-{self.hex}.tmp", f".rt-{self.hex}.tmp.old", "заметка.md"):
            touch(self.root / name)
        touch(self.data / "sub" / f"state.sqlite3.incomplete-{uuid.uuid4().hex}")         # в папке данных — не вглубь
        report = leftovers.find(self.db, [self.root])
        found = self.found(report)
        self.assertEqual(len(found), 5, sorted(found))
        self.assertEqual(report["count"], 5)
        self.assertEqual(report["younger"], 2)
        self.assertEqual(found[copy.name]["kind"], "database_copy")
        self.assertEqual(found[copy.name]["size"], 100)
        self.assertFalse(found[copy.name]["second_name_of_database"])
        self.assertEqual(found[copy.name]["hint"], leftovers.HINT_COPY)
        self.assertEqual(found[f"{copy.name}-wal"]["hint"], leftovers.HINT_COPY_SIDE)
        self.assertEqual(found[temp.name]["kind"], "document_temp")
        self.assertGreaterEqual(found[temp.name]["age_days"], 2.9)
        self.assertEqual(report["bytes"], sum(item["size"] for item in report["items"]))
        self.assertIn(str(self.data / "db-backups"), report["checked"])

    def test_nothing_is_opened_changed_or_removed(self):
        touch(self.data / f"state.sqlite3.incomplete-{self.hex}", b"SQLite format 3\x00" + b"\x00" * 100)
        touch(self.root / f".rt-{self.hex}.tmp")
        before = tree_state(self.base)
        with mock.patch.object(sqlite3, "connect", side_effect=AssertionError("файл открыт SQLite")):
            report = leftovers.find(self.db, [self.root])
        self.assertEqual(report["count"], 2)
        self.assertEqual(tree_state(self.base), before)          # и -wal/-shm у копии не появились

    def test_second_name_of_working_database_is_marked_even_if_fresh(self):
        second = self.data / f"state.sqlite3.incomplete-{self.hex}"
        os.link(self.db, second)                                 # как после восстановления, если имя не убралось
        other = self.base / "other.sqlite3"
        other.write_bytes(b"x")
        linked = self.data / f"r.sqlite3.incomplete-{uuid.uuid4().hex}"
        os.link(other, linked)
        os.utime(linked, (OLD, OLD))
        found = self.found(leftovers.find(self.db, []))
        self.assertTrue(found[second.name]["second_name_of_database"])
        self.assertEqual(found[second.name]["hard_links"], 2)
        self.assertEqual(found[second.name]["hint"], leftovers.HINT_SECOND_NAME)
        self.assertFalse(found[linked.name]["second_name_of_database"])
        self.assertEqual(found[linked.name]["hint"], leftovers.HINT_LINKED)

    def test_links_are_not_followed(self):
        outside = self.base / "снаружи"
        touch(outside / f".rt-{self.hex}.tmp")
        try:
            make_link_to_folder(outside, self.root / "ссылка")
        except OSError as ex:
            self.skipTest(f"ссылку на папку здесь не создать: {ex}")
        report = leftovers.find(self.db, [self.root])
        self.assertEqual(report["count"], 0)
        # корень, который сам ссылка, не обходится
        link_root = self.base / "корень-ссылка"
        make_link_to_folder(outside, link_root)
        report = leftovers.find(self.db, [link_root])
        self.assertEqual(report["count"], 0)
        self.assertEqual(report["skipped"][0]["path"], str(link_root))

    def test_missing_folders_and_entry_limit(self):
        report = leftovers.find(self.db, [self.base / "нет такой"])
        self.assertEqual((report["count"], report["skipped"]), (0, []))
        for _ in range(5):
            touch(self.root / f".rt-{uuid.uuid4().hex}.tmp")
        with mock.patch.object(leftovers, "MAX_ENTRIES", 3):
            report = leftovers.find(self.db, [self.root])
        self.assertTrue(report["truncated"])
        self.assertLessEqual(report["count"], 3)


class LeftoversReport(unittest.TestCase):
    """Сводка программы: корни документов — из базы; самопроверка и экран показывают тот же список."""

    def setUp(self):
        self.base = Path(tempfile.mkdtemp(prefix="leftovers-report-", dir=os.environ.get("TEMP")))
        self.addCleanup(shutil.rmtree, self.base, True)
        self.db = self.base / "state.sqlite3"
        patcher = mock.patch.object(paths, "STATE_DB", self.db)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.root = self.base / "Записи" / "a.wav.rus-transcriber"
        self.root.mkdir(parents=True)
        store = storage.initialize(self.db, initial_display_name="Тест")
        try:
            store.register_root(self.root)
        finally:
            store.close()
        self.temp = touch(self.root / f".rt-{uuid.uuid4().hex}.tmp", b"abc")
        self.copy = touch(self.base / f"state.sqlite3.incomplete-{uuid.uuid4().hex}", b"c" * 10)

    def test_report_reads_roots_from_database(self):
        report = core.leftovers_report()
        # корни в базе хранятся в виде os.path.normcase (в Windows — в нижнем регистре)
        self.assertEqual(sorted(os.path.normcase(item["path"]) for item in report["items"]),
                         sorted(os.path.normcase(str(path)) for path in (self.temp, self.copy)))
        self.assertNotIn("note", report)
        self.assertTrue(self.temp.exists() and self.copy.exists())

    def test_unreadable_database_still_checks_its_folders(self):
        with mock.patch.object(recordings, "open_store", side_effect=storage.StorageError("cannot open database")):
            report = core.leftovers_report()
        self.assertEqual([item["path"] for item in report["items"]], [str(self.copy)])
        self.assertIn("корни документов не проверены", report["note"])

    def test_self_check_part_never_fails(self):
        self.assertEqual(core.leftovers_summary()["count"], 2)
        with mock.patch.object(core, "leftovers_report", side_effect=OSError("synthetic")):
            self.assertIn("synthetic", core.leftovers_summary()["error"])

    def test_screen_route_returns_the_list(self):
        import workflow_page
        srv, url, _ = workflow_page.make_server(8790)
        self.addCleanup(srv.server_close)
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(srv.shutdown)
        host = f"127.0.0.1:{srv.server_port}"
        conn = http.client.HTTPConnection(host, timeout=10)
        try:
            conn.request("GET", urlsplit(url).path + "api/leftovers", headers={"Host": host})
            response = conn.getresponse()
            body = json.loads(response.read().decode("utf-8"))
        finally:
            conn.close()
        self.assertEqual(response.status, 200)
        self.assertEqual(body["count"], 2)
        html = (Path(workflow_page.__file__).with_name("workflow.html")).read_text(encoding="utf-8")
        self.assertIn('call("api/leftovers")', html)


if __name__ == "__main__":
    unittest.main()
