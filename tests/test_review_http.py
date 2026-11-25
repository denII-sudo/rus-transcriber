"""HTTP-защита локальных карточек на синтетическом пакете, без моделей и реальных данных."""
import http.client
import json
import os
from pathlib import Path
import select
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from urllib.parse import quote, urlsplit


PROJECT = Path(__file__).resolve().parent.parent


class ReviewHTTP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="review_http_")
        root = Path(cls.tmp.name)
        cls.pack = root / "demo.review"
        (cls.pack / "clips").mkdir(parents=True)
        (cls.pack / "clips" / "r1.mp3").write_bytes(b"demo audio")
        (cls.pack / "manifest.json").write_text(json.dumps({
            "version": "v1", "audio": "demo.wav", "items": [{"id": "r1", "t": "00:00:01",
                "clip": "clips/r1.mp3", "text": "да", "speaker": "Участник", "t1": 0.0, "t2": 0.0,
                "question": "Как правильно?", "diagnostic_id": "missed-0001",
                "options": [{"label": "да", "decision": {}}]}]
        }), encoding="utf-8")
        (cls.pack / "answers.old.json").write_text('{"archive": true}', encoding="utf-8")
        (cls.pack / "cloud.json").write_text('{"debug": true}', encoding="utf-8")
        cls.status = root / "status.json"
        data, temp = root / "data", root / "temp"
        data.mkdir()
        temp.mkdir()
        env = dict(os.environ, RUS_TRANSCRIBER_DATA=str(data), TEMP=str(temp), TMP=str(temp),
                   PYTHONUTF8="1", PYTHONDONTWRITEBYTECODE="1")
        # Голосовой маршрут проверяем без загрузки модели.
        code = ("import sys, review_page; "
                "review_page.P.recognize_here=lambda files: {str(p): 'первый' for p in files}; "
                "sys.argv=['review_page.py', sys.argv[1], '--status', sys.argv[2], '--markdown', sys.argv[3], '--port', '19350']; "
                "review_page.main()")
        cls.proc = subprocess.Popen([sys.executable, "-B", "-c", code, str(root / "demo.transcript.json"),
                                     str(cls.status), str(root / "demo.transcript.md")], cwd=str(PROJECT), env=env,
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

    def request(self, path="", method="GET", body=None, headers=None, absolute=False):
        conn = http.client.HTTPConnection("127.0.0.1", self.url.port, timeout=10)
        try:
            conn.request(method, path if absolute else self.url.path + path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            conn.close()

    def post(self, path, body, **headers):
        return self.request(path, "POST", body, {"Origin": f"http://127.0.0.1:{self.url.port}", **headers})

    def test_page_and_debug_files_require_session(self):
        status, page, headers = self.request()
        self.assertEqual(status, 200)
        self.assertIn(b"api/answers", page)
        self.assertEqual(headers["Referrer-Policy"], "no-referrer")
        for name in ("manifest.json", "clips/r1.mp3", "answers.old.json", "cloud.json", "api/answers"):
            with self.subTest(name=name):
                self.assertEqual(self.request(name)[0], 200)
                self.assertEqual(self.request("/" + name, absolute=True)[0], 403)

    def test_answers_name_heard_places(self):
        """Раздел 10: ответ api/answers называет места, уже прослушанные целиком (heard_ids), — список."""
        self.assertIsInstance(json.loads(self.request("api/answers")[1])["heard_ids"], list)

    def test_host_origin_token_and_content_type(self):
        payload = json.dumps({"id": "r1", "version": "v1", "option": 0}).encode()
        origin = f"http://127.0.0.1:{self.url.port}"
        self.assertEqual(self.request("api/answers", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.request("api/answer", "POST", payload,
                         {"Origin": "https://evil.example", "Content-Type": "text/plain"})[0], 403)
        self.assertEqual(self.request("api/answer", "POST", payload,
                         {"Content-Type": "application/json"})[0], 403)
        self.assertEqual(self.post("api/answer", payload, **{"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(self.request("/api/answer", "POST", payload,
                         {"Origin": origin, "Content-Type": "application/json"}, absolute=True)[0], 403)
        status, data, _ = self.post("api/answer", payload, **{"Content-Type": "application/json"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["accepted"][0]["id"], "r1")

    def test_voice_rejects_traversal_before_write_and_accepts_valid_card(self):
        suffix = "?id=" + quote(r"..\..\escape", safe="") + "&version=v1"
        self.assertEqual(self.post("api/voice" + suffix, b"demo", **{"Content-Type": "audio/wav"})[0], 400)
        self.assertEqual(list(Path(self.tmp.name).glob("escape*")), [])
        self.assertEqual(self.post("api/voice?id=r1&version=old", b"demo",
                                  **{"Content-Type": "audio/wav"})[0], 400)
        before = self.request("api/answers")[1]
        status, data, _ = self.post("api/voice?id=r1&version=v1", b"demo",
                                     **{"Content-Type": "audio/wav"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(data)["heard"], "первый")
        self.assertEqual(json.loads(data)["match"], {"kind": "option", "option": 0})
        self.assertEqual(self.request("api/answers")[1], before)
        answer = json.dumps({"id": "r1", "version": "v1", "text": "первый", "heard": True}).encode()
        self.assertEqual(self.post("api/answer", answer, **{"Content-Type": "application/json"})[0], 200)
        store = json.loads(self.request("api/answers")[1])
        self.assertEqual(store["answers"]["r1"]["kind"], "option")
        self.assertEqual(store["answers"]["r1"]["diagnostic_id"], "missed-0001")
        self.assertEqual(len(list((self.pack / "voice").glob("page-*.wav"))), 1)

    def test_interaction_events_are_versioned_and_linked_to_diagnostic(self):
        def send(event_type, version="v1"):
            payload = json.dumps({"id": "r1", "version": version, "type": event_type,
                                  "session": "session-1234"}).encode()
            return self.post("api/event", payload, **{"Content-Type": "application/json"})

        status, body, _ = send("card_visible")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["stored"])
        status, body, _ = send("card_visible")
        self.assertEqual(status, 200)
        self.assertTrue(json.loads(body)["duplicate"])
        self.assertEqual(send("audio_play")[0], 200)
        self.assertEqual(send("audio_ended")[0], 200)
        self.assertEqual(send("audio_play", "old")[0], 400)
        self.assertEqual(send("unexpected")[0], 400)
        store = json.loads(self.request("api/answers")[1])
        events = store["interaction_events"]
        self.assertEqual([e["type"] for e in events], ["card_visible", "audio_play", "audio_ended"])
        self.assertTrue(all(e["diagnostic_id"] == "missed-0001" for e in events))

    def test_limits(self):
        from review_page import MAX_ANSWER_BYTES, MAX_VOICE_BYTES
        self.assertEqual(self.post("api/answer", b"x" * (MAX_ANSWER_BYTES + 1),
                                   **{"Content-Type": "application/json"})[0], 413)
        # Не отправляем большой массив: сервер обязан отклонить по заголовку до чтения тела.
        conn = http.client.HTTPConnection("127.0.0.1", self.url.port, timeout=10)
        try:
            conn.putrequest("POST", self.url.path + "api/voice?id=r1&version=v1")
            conn.putheader("Origin", f"http://127.0.0.1:{self.url.port}")
            conn.putheader("Content-Type", "audio/wav")
            conn.putheader("Content-Length", str(MAX_VOICE_BYTES + 1))
            conn.endheaders()
            self.assertEqual(conn.getresponse().status, 413)
        finally:
            conn.close()

    def raw_head(self, path, length, headers=None, target=None):
        port = self.url.port
        fields = {"Host": f"127.0.0.1:{port}", "Origin": f"http://127.0.0.1:{port}", "Content-Type": "application/json",
                  "Content-Length": str(length), "Connection": "close"}
        fields.update(headers or {})
        return (f"POST {target or self.url.path + path} HTTP/1.1\r\n"
                + "".join(f"{k}: {v}\r\n" for k, v in fields.items()) + "\r\n").encode()

    def raw_post(self, path, length, parts, pause=0.1, headers=None, target=None):
        """Запрос на сыром сокете: заголовки с Content-Length=length и первая часть тела, после каждой паузы — следующая;
        затем ответ читается до закрытия соединения. -> (статус, заголовки в нижнем регистре, тело, секунд до конца ответа)."""
        port = self.url.port
        head = self.raw_head(path, length, headers, target)
        started = time.time()
        with socket.create_connection(("127.0.0.1", port), timeout=10) as sock:
            sock.sendall(head + parts[0])
            for part in parts[1:]:
                time.sleep(pause)
                sock.sendall(part)
            data = b""
            while True:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                data += chunk
        top, _, body = data.partition(b"\r\n\r\n")
        lines = top.decode("iso-8859-1").split("\r\n")
        headers = dict(line.lower().split(": ", 1) for line in lines[1:])
        return int(lines[0].split()[1]), headers, body, time.time() - started

    def test_413_for_a_body_that_arrives_after_the_refusal_still_reaches_the_client(self):
        # Раньше сервер отвечал 413 по заголовку и закрывал соединение, пока тело ещё шло: в Windows клиент получал
        # сброс (WinError 10053) вместо ответа. Теперь тело до MAX_DRAIN_BYTES дочитывается, потом идёт 413.
        from review_page import MAX_ANSWER_BYTES
        length = MAX_ANSWER_BYTES + 1
        status, headers, body, _ = self.raw_post("api/answer", length, [b"x" * 100, b"x" * (length - 100)])
        self.assertEqual(status, 413)
        self.assertEqual(headers["connection"], "close")
        self.assertIn("недопустимый размер", json.loads(body)["error"])

    def test_every_early_refusal_reads_the_body_first_so_the_client_gets_the_answer(self):
        # Отказы до чтения тела (403 по адресу, источнику и сессии, 404, 415, 413): тело приходит уже после отказа.
        # Без дочитывания в Windows клиент получал сброс соединения вместо ответа (CI №201: 403 на api/event).
        from review_page import MAX_ANSWER_BYTES
        body = b'{"id": "r1", "version": "v", "text": "x"}'
        wrong_session = self.url.path.rstrip("/") + "x/api/event"
        cases = (
            ("403 адрес", 403, {"path": "api/event", "headers": {"Host": "evil.example"}}),
            ("403 источник", 403, {"path": "api/event", "headers": {"Origin": "https://evil.example"}}),
            ("403 сессия", 403, {"path": "", "target": wrong_session}),
            ("404 маршрут", 404, {"path": "api/nothing"}),
            ("415 тип", 415, {"path": "api/event", "headers": {"Content-Type": "text/plain"}}),
            ("413 размер", 413, {"path": "api/answer", "length": MAX_ANSWER_BYTES + 1}),
        )
        for name, code, case in cases:
            with self.subTest(name):
                length = case.get("length", len(body))
                parts = [body[:5], body[5:]] if length == len(body) else [b"x" * 100, b"x" * (length - 100)]
                status, headers, answer, _ = self.raw_post(case["path"], length, parts, headers=case.get("headers"),
                                                           target=case.get("target"))
                self.assertEqual(status, code)
                self.assertEqual(headers["connection"], "close")
                self.assertIn("error", json.loads(answer))

    def test_a_body_that_trickles_in_does_not_hold_the_refusal_longer_than_the_drain_timeout(self):
        # DRAIN_TIMEOUT — общий срок на всё дочитывание, а не на каждое чтение: байт раз в полсекунды не продлевает его
        from review_page import DRAIN_TIMEOUT, MAX_ANSWER_BYTES
        started = time.time()
        status = None
        with socket.create_connection(("127.0.0.1", self.url.port), timeout=10) as sock:
            sock.sendall(self.raw_head("api/answer", MAX_ANSWER_BYTES + 1))
            while time.time() - started < DRAIN_TIMEOUT + 5:
                if select.select([sock], [], [], 0.5)[0]:
                    try:
                        data = sock.recv(65536)
                    except OSError:
                        data = b""                       # соединение закрыто с непрочитанным: в Windows это сброс
                    status = int(data.split()[1]) if data.startswith(b"HTTP/") else None
                    break
                try:
                    sock.sendall(b"x")
                except OSError:
                    break
            seconds = time.time() - started
        self.assertGreaterEqual(seconds, DRAIN_TIMEOUT - 0.5)
        self.assertLess(seconds, DRAIN_TIMEOUT + 1.5)             # а не DRAIN_TIMEOUT + 5: сервер перестал ждать
        self.assertIn(status, (413, None))                        # ответ, если соединение его ещё принимает

    def test_body_over_the_drain_limit_is_refused_at_once_and_not_read(self):
        from review_page import MAX_DRAIN_BYTES
        status, headers, _, seconds = self.raw_post("api/answer", MAX_DRAIN_BYTES + 1, [b""])
        self.assertEqual((status, headers["connection"]), (413, "close"))
        self.assertLess(seconds, 1.5)                         # тело, которого никто не ждёт, сервер не дожидается

    def test_declared_body_that_never_comes_gets_413_after_the_drain_timeout(self):
        from review_page import DRAIN_TIMEOUT, MAX_ANSWER_BYTES
        status, headers, _, seconds = self.raw_post("api/answer", MAX_ANSWER_BYTES + 1, [b""])
        self.assertEqual((status, headers["connection"]), (413, "close"))
        self.assertGreaterEqual(seconds, DRAIN_TIMEOUT - 0.5)
        self.assertLess(seconds, DRAIN_TIMEOUT + 6)           # сервер не висит, пока клиент молчит

    def test_new_routes_enforce_origin_and_version(self):
        payload = json.dumps({"id": "r1", "version": "old", "text": "пример"}).encode()
        for route in ("api/preview", "api/apply", "api/event"):
            with self.subTest(route=route):
                self.assertEqual(self.request(route, "POST", payload,
                                 {"Origin": "https://evil.example", "Content-Type": "application/json"})[0], 403)
                self.assertEqual(self.post(route, payload, **{"Content-Type": "text/plain"})[0], 415)
                self.assertEqual(self.post(route, payload, **{"Content-Type": "application/json"})[0], 400)

    def test_end_to_end_apply_and_undo(self):
        root = Path(self.tmp.name)
        jp, md = root / "demo.transcript.json", root / "demo.transcript.md"
        jp.write_text(json.dumps({"file": "demo.wav", "duration": "00:00:01", "speakers": ["Участник"],
            "segments": [{"speaker": "Участник", "label": "Участник", "start": 0.0, "end": 0.2,
                "text": "да", "words": [{"w": "да", "s": 0.0, "e": 0.2, "c": 0.9}]}],
            "review": [{"id": "r1", "t": 0.0, "t1": 0.0, "t2": 0.0, "speaker": "Участник",
                        "text": "да", "status": "open", "reasons": []}], "edits": []}), encoding="utf-8")
        md.write_text("demo", encoding="utf-8")
        headers = {"Content-Type": "application/json"}
        def post(route, obj):
            code, body, _ = self.post(route, json.dumps(obj).encode(), **headers)
            self.assertEqual(code, 200, body)
            return json.loads(body)
        preview = post("api/preview", {"id": "r1", "version": "v1", "text": "да"})
        self.assertIn("cards", preview)
        post("api/answer", {"id": "r1", "version": "v1", "option": 0})
        result = post("api/apply", {"version": "v1"})
        self.assertEqual(result["review"]["successful_ids"], ["r1"])
        self.assertNotIn("voice_profile_warning", result)
        self.assertEqual(json.loads(jp.read_text(encoding="utf-8"))["review"][0]["status"], "resolved")
        post("api/answer", {"id": "r1", "version": "v1", "kind": "revert"})
        result = post("api/apply", {"version": "v1"})
        self.assertEqual(result["reverted"], ["r1"])
        self.assertEqual(result["voice_profile_warning"]["answer_ids"], ["r1"])
        self.assertIn("мог остаться", result["voice_profile_warning"]["message"])
        self.assertEqual(json.loads(jp.read_text(encoding="utf-8"))["review"][0]["status"], "open")

    def test_preview_refuses_conflicting_saved_answer_without_writing(self):
        root = Path(self.tmp.name)
        jp, md = root / "demo.transcript.json", root / "demo.transcript.md"
        mf, af = self.pack / "manifest.json", self.pack / "answers.json"
        files = (jp, md, mf, af)
        saved = {p: p.read_bytes() if p.exists() else None for p in files}
        try:
            reviews = [{"id": rid, "t": 0.0, "t1": 0.0, "t2": 0.0, "speaker": "Участник",
                        "text": "да", "status": "open", "reasons": []} for rid in ("r1", "r2")]
            jp.write_text(json.dumps({"file": "demo.wav", "duration": "00:00:01", "recording_id": "demo",
                "speakers": ["Участник"], "segments": [{"speaker": "Участник", "label": "Участник",
                "start": 0.0, "end": 0.2, "text": "да", "words": [{"w": "да", "s": 0.0, "e": 0.2,
                "c": 0.9, "word_id": "w1"}]}], "review": reviews, "edits": []}), encoding="utf-8")
            md.write_text("demo", encoding="utf-8")
            mf.write_text(json.dumps({"version": "v1", "recording_id": "demo", "items": [
                dict(r, word_ids=["w1"], options=[{"label": "да", "decision": {}}]) for r in reviews]}), encoding="utf-8")
            af.write_text(json.dumps({"version": "v1", "answers": {"r2": {"id": "r2", "version": "v1",
                "kind": "free", "text": "второе", "source": "local", "ts": "2026-01-01"}}}), encoding="utf-8")
            before = {p: p.read_bytes() for p in files}
            payload = json.dumps({"id": "r1", "version": "v1", "text": "первое"}).encode()
            status, body, _ = self.post("api/preview", payload, **{"Content-Type": "application/json"})
            self.assertEqual(status, 200, body)
            self.assertEqual(json.loads(body)["preflight_conflicts"][0]["cards"], ["r1", "r2"])
            self.assertEqual({p: p.read_bytes() for p in files}, before)
        finally:
            for p, original in saved.items():
                if original is None:
                    p.unlink(missing_ok=True)
                else:
                    p.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
