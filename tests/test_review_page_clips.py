"""Звук карточек перенесённых записей (разбор 2026-11-26, приватный репозиторий: research/2026-11-26-cards-audio).

Перенос П7 клипы не копирует (Р-3): у старых карточек ссылка clips/<id>.mp3 указывает в новую папку пакета, а файл
остался в прежней (pack_dir(legacy_job_json)). Страница отдаёт его оттуда — только чтение и только клип карточки или
её истории из текущего manifest. Range (206) — чтобы клип можно было перематывать в Chromium. Ошибка загрузки звука
показывается на карточке надписью. Данные синтетические, всё во временной папке.
"""
import atexit
import http.client
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

PROJECT = Path(__file__).resolve().parent.parent
if "paths" not in sys.modules:          # временная папка данных — до импорта программы
    os.environ["RUS_TRANSCRIBER_DATA"] = tempfile.mkdtemp(prefix="rt-page-clips-data-")
    atexit.register(shutil.rmtree, os.environ["RUS_TRANSCRIBER_DATA"], ignore_errors=True)
sys.path.insert(0, str(PROJECT))

import review_page  # noqa: E402

OLD_R1, OLD_R2, NEW_R2 = b"old clip r1 " * 20, b"old clip r2", b"new clip r2 rev 2"


def item(aid, clip, history=()):
    return {"id": aid, "t": "00:00:01", "clip": clip, "text": "да", "speaker": "Участник", "t1": 1.0, "t2": 1.5,
            "question": "Как правильно?", "options": [{"label": "да", "decision": {}}],
            "history": [{"rev": 1, "clip": h} for h in history]}


def migrated(root):
    """Старая папка пакета (рядом со старой стенограммой) и новая папка пакета перенесённой записи."""
    old_json = root / "Старая папка (1)" / "встреча.v2.transcript.json"
    old = old_json.with_name("встреча.v2.review")
    (old / "clips").mkdir(parents=True)
    (old / "clips" / "r1.mp3").write_bytes(OLD_R1)
    (old / "clips" / "r2.mp3").write_bytes(OLD_R2)
    (old / "clips" / "r2.r2.mp3").write_bytes(b"old folder copy must not win")
    (old / "clips" / "extra.mp3").write_bytes(b"not in manifest")
    (old / "clips" / "note.txt").write_bytes(b"not mp3")
    (old / "secret.mp3").write_bytes(b"outside clips")
    try:                                 # ссылка из clips наружу: путь без «..», но файл вне clips
        import _winapi
        _winapi.CreateJunction(str(old), str(old / "clips" / "link"))
    except (ImportError, OSError):
        pass
    new_json = root / "Новая папка" / "v4" / "встреча.v2.transcript.json"
    new = new_json.with_name("встреча.v2.review")
    (new / "clips").mkdir(parents=True)
    (new / "clips" / "r2.r2.mp3").write_bytes(NEW_R2)
    manifest = {"version": "v1", "audio": "встреча.mp3", "legacy_job_json": str(old_json), "clips_dir": "clips",
                "items": [item("r1", "clips/r1.mp3"), item("r2", "clips/r2.r2.mp3", ["clips/r2.mp3"]),
                          item("r3", "clips/r3.mp3"), item("r4", "clips/note.txt"),
                          item("r5", "clips/../secret.mp3"), item("r6", "clips/x/../r1.mp3"),
                          item("r7", "clips/link/secret.mp3")]}
    (new / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False), encoding="utf-8")
    return new_json, manifest


class LegacyClipsHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="review_page_clips_")
        root = Path(cls.tmp.name)
        jp, _ = migrated(root)
        cls.status = root / "status.json"
        data, temp = root / "data", root / "temp"
        data.mkdir()
        temp.mkdir()
        env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
                   PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        cls.proc = subprocess.Popen([sys.executable, "-B", str(PROJECT / "review_page.py"), str(jp), "--status",
                                     str(cls.status), "--port", "19380"], cwd=str(PROJECT), env=env,
                                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            if cls.status.exists():
                try:
                    url = json.loads(cls.status.read_text(encoding="utf-8")).get("url")
                    if url:
                        cls.url = urlsplit(url)
                        return
                except (OSError, ValueError):
                    pass
            if cls.proc.poll() is not None:
                break
            time.sleep(0.1)
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()
        raise RuntimeError("тестовая страница не запустилась")

    @classmethod
    def tearDownClass(cls):
        cls.proc.terminate()
        cls.proc.wait(timeout=10)
        cls.tmp.cleanup()

    def get(self, path, headers=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.url.port, timeout=10)
        try:
            conn.request("GET", self.url.path + path, headers=headers or {})
            r = conn.getresponse()
            return r.status, r.read(), dict(r.getheaders())
        finally:
            conn.close()

    def test_old_card_clip_comes_from_the_previous_pack_folder(self):
        code, body, headers = self.get("clips/r1.mp3")
        self.assertEqual((code, body), (200, OLD_R1))
        self.assertEqual(headers["Content-Type"], "audio/mpeg")
        self.assertEqual(headers["Accept-Ranges"], "bytes")
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.get("clips/r2.mp3")[:2], (200, OLD_R2), "клип прежней редакции (history)")

    def test_new_pack_folder_wins(self):
        self.assertEqual(self.get("clips/r2.r2.mp3")[:2], (200, NEW_R2))

    def test_only_listed_mp3_inside_the_old_clips_folder(self):
        for path in ("clips/extra.mp3",           # файл есть, но карточки с ним нет
                     "clips/r3.mp3",              # карточка есть, файла нет нигде
                     "clips/note.txt",            # в manifest, но не mp3
                     "clips/../secret.mp3",       # в manifest, но выходит из clips
                     "clips/%2e%2e/secret.mp3",
                     "clips/x/../r1.mp3",         # в manifest, «..» внутри clips — тоже нет
                     "clips/link/secret.mp3",     # через ссылку наружу (если её удалось создать)
                     "clips%5cr1.mp3",
                     "answers.json"):
            with self.subTest(path=path):
                self.assertEqual(self.get(path)[0], 404)

    def test_range_gives_part_of_the_clip(self):
        n = len(OLD_R1)
        code, body, headers = self.get("clips/r1.mp3", {"Range": "bytes=2-5"})
        self.assertEqual((code, body, headers["Content-Range"]), (206, OLD_R1[2:6], f"bytes 2-5/{n}"))
        self.assertEqual(headers["Content-Length"], "4")
        code, body, headers = self.get("clips/r1.mp3", {"Range": "bytes=-3"})
        self.assertEqual((code, body, headers["Content-Range"]), (206, OLD_R1[-3:], f"bytes {n - 3}-{n - 1}/{n}"))
        code, body, headers = self.get("clips/r1.mp3", {"Range": "bytes=10-"})
        self.assertEqual((code, body), (206, OLD_R1[10:]))
        code, body, headers = self.get("clips/r2.r2.mp3", {"Range": "bytes=0-"})
        self.assertEqual((code, body, headers["Content-Range"]), (206, NEW_R2, f"bytes 0-{len(NEW_R2) - 1}/{len(NEW_R2)}"))
        code, body, headers = self.get("clips/r1.mp3", {"Range": f"bytes={n}-"})
        self.assertEqual((code, body, headers["Content-Range"]), (416, b"", f"bytes */{n}"))
        for odd in ("bytes=5-2", "bytes=0-1,4-5", "items=0-1", "bytes=-"):
            with self.subTest(range=odd):
                self.assertEqual(self.get("clips/r1.mp3", {"Range": odd})[:2], (200, OLD_R1))


class LegacyClipRules(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix="review_page_clips_unit_")
        self.addCleanup(tmp.cleanup)
        self.jp, self.manifest = migrated(Path(tmp.name))

    def clip(self, rel, manifest=None):
        with mock.patch.object(review_page.P, "manifest", return_value=self.manifest if manifest is None else manifest):
            return review_page.legacy_clip(self.jp, rel)

    def test_listed_old_clip(self):
        self.assertEqual(self.clip("clips/r1.mp3").read_bytes(), OLD_R1)

    def test_no_legacy_job_json_no_fallback(self):
        """Запись базы, созданная не переносом, и файловый пакет: прежнее поведение — 404."""
        self.assertIsNone(self.clip("clips/r1.mp3", dict(self.manifest, legacy_job_json=None)))
        self.assertIsNone(self.clip("clips/r1.mp3", {k: v for k, v in self.manifest.items() if k != "legacy_job_json"}))
        self.assertIsNone(self.clip("clips/r1.mp3", {}))

    def test_old_folder_gone(self):
        shutil.rmtree(Path(self.manifest["legacy_job_json"]).with_name("встреча.v2.review"))
        self.assertIsNone(self.clip("clips/r1.mp3"))

    def test_refusals(self):
        for rel in ("clips/extra.mp3", "clips/note.txt", "clips/../secret.mp3", "clips/%2E%2E/secret.mp3",
                    "clips\\r1.mp3", "/clips/r1.mp3", "r1.mp3", "clips/r3.mp3",
                    "clips/x/../r1.mp3"):         # «..» запрещено, даже если путь остаётся внутри clips
            with self.subTest(rel=rel):
                self.assertIsNone(self.clip(rel))

    def test_link_out_of_clips_is_refused(self):
        """Путь без «..», но через ссылку (junction) ведёт из clips наружу: проверка после resolve()."""
        link = Path(self.manifest["legacy_job_json"]).with_name("встреча.v2.review") / "clips" / "link"
        if not (link / "secret.mp3").is_file():
            self.skipTest("ссылку (junction) создать не удалось")
        self.assertIsNone(self.clip("clips/link/secret.mp3"))

    def test_byte_range(self):
        cases = {None: None, "": None, "bytes=0-": (0, 9), "bytes=2-4": (2, 4), "bytes=8-100": (8, 9),
                 "bytes=-3": (7, 9), "bytes=-30": (0, 9), "bytes=10-": False, "bytes=-0": False,
                 "bytes=5-2": None, "bytes=0-1,3-4": None, "bytes=-": None, "items=0-1": None,
                 "bytes=" + "9" * 16 + "-": None}
        for header, want in cases.items():
            with self.subTest(header=header):
                self.assertEqual(review_page.byte_range(header, 10), want)
        self.assertIs(review_page.byte_range("bytes=0-", 0), False)


class ClipErrorOnPage(unittest.TestCase):
    @unittest.skipUnless(shutil.which("node"), "для проверки JavaScript нужен Node.js")
    def test_failed_clip_is_replaced_by_a_note(self):
        """Ошибка загрузки звука: вместо молча посеревшего плеера на карточке надпись (cards.html)."""
        html = (PROJECT / "cards.html").read_text(encoding="utf-8")
        const = next(line for line in html.splitlines() if line.strip().startswith("const CLIP_FAILED"))
        block = "    if (!man.clips_removed && it.clip) {" + html.split(
            "    if (!man.clips_removed && it.clip) {", 1)[1].split("      c.append(au);\n    }\n", 1)[0] \
            + "      c.append(au);\n    }\n"
        script = const + r"""
          const made = [];
          const document = {createElement: tag => { const el = {tag, handlers: {}, addEventListener(t, f) {
            (this.handlers[t] = this.handlers[t] || []).push(f); }, replaceWith(x) { this.replacedBy = x; }};
            made.push(el); return el; }};
          const c = {children: [], append(x) { this.children.push(x); }};
          const man = {}, it = {id: "r1", clip: "clips/r1.mp3"};
          const sendInteraction = () => {}, trackPlayed = () => {};
        """ + block + r"""
          const au = c.children[0];
          if (au.tag !== "audio" || au.src !== "clips/r1.mp3") throw Error("плеера нет");
          if (!au.handlers.error) throw Error("нет обработчика error");
          au.handlers.error.forEach(f => f());
          if (!au.replacedBy || au.replacedBy.textContent !== CLIP_FAILED) throw Error("надписи нет");
          if (!CLIP_FAILED.includes("не загрузился")) throw Error("текст надписи");
        """
        result = subprocess.run([shutil.which("node"), "-e", "(() => {" + script + "})();"], capture_output=True,
                                text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
