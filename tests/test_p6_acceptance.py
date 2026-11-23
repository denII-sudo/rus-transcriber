"""П6: синтетическая проверка поставки и воспроизводимый профиль масштаба."""
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import tracemalloc
import unittest
import uuid
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock
from urllib.parse import urlsplit

import core
import documents
import quotes
import paths
import rt
import storage
import workflow
import workflow_page


PROJECT = Path(__file__).resolve().parents[1]
INSTALLER = PROJECT / "install.ps1"
REQUIRED_RUNTIME = {
    "storage.py", "documents.py", "recordings.py", "workflow.py",
    "product_policy.py", "model_tasks.py", "llm_provider.py",
    "workflow_page.py", "workflow.html", "rt.py", "answer_phrases.tsv",
}
# файл-метка изолированной папки приёмки, без него install.ps1 -AcceptanceCopyOnly отказывает
SANDBOX_MARKER = "rus-transcriber-acceptance.sandbox"


class P6InstallUpdateTests(unittest.TestCase):
    """Папки создаются только во временной папке (TEMP): тест не зависит от конкретного компьютера."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="p6-install-")
        self.root = Path(self.temp.name)
        (self.root / SANDBOX_MARKER).write_text("synthetic acceptance sandbox\n", encoding="utf-8")
        self.target = self.root / "program"
        self.data = self.root / "data"
        self.target.mkdir()
        self.data.mkdir()

    def tearDown(self):
        self.temp.cleanup()

    def unmarked_dir(self):
        """Соседняя временная папка без метки: для проверок отказа."""
        other = tempfile.TemporaryDirectory(prefix="p6-outside-")
        self.addCleanup(other.cleanup)
        return Path(other.name)

    def assert_refused(self, result, *paths):
        self.assertNotEqual(result.returncode, 0, result.stdout.decode("utf-8", "replace"))
        for path in paths:
            self.assertFalse((path / "server.py").exists(), path)

    def run_copy_only_installer(self, *, enable_sandbox=True, target=None, data=None):
        executable = shutil.which("powershell.exe") or shutil.which("powershell")
        if not executable:
            self.skipTest("Windows PowerShell is unavailable")
        env = dict(os.environ)
        if enable_sandbox:
            env["RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX"] = "1"
        else:
            env.pop("RUS_TRANSCRIBER_ACCEPTANCE_SANDBOX", None)
        return subprocess.run(
            [executable, "-NoLogo", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File", str(INSTALLER),
             "-Target", str(target or self.target), "-DataDir", str(data or self.data),
             "-SkipClaudeConfig", "-AcceptanceCopyOnly"],
            cwd=PROJECT, env=env, capture_output=True, timeout=60,
        )

    def test_runtime_manifest_and_sandbox_update_preserve_database(self):
        script = INSTALLER.read_text(encoding="utf-8-sig")
        match = re.search(r'foreach \(\$f in @\((.*?)\)\) \{', script, re.S)
        self.assertIsNotNone(match, "runtime copy list not found in installer")
        package_files = set(re.findall(r'"([^"]+)"', match.group(1)))
        self.assertTrue(REQUIRED_RUNTIME <= package_files)

        db_path = self.data / "state.sqlite3"
        store = storage.initialize(db_path, initial_display_name="synthetic")
        store.close()
        before = hashlib.sha256(db_path.read_bytes()).hexdigest()
        marker = self.data / "user-data-sentinel.txt"
        marker.write_text("preserve", encoding="utf-8")

        refused = self.run_copy_only_installer(enable_sandbox=False)
        self.assert_refused(refused, self.target)
        outside = self.unmarked_dir() / "program"
        refused_outside = self.run_copy_only_installer(target=outside)
        self.assert_refused(refused_outside, outside)

        installed = self.run_copy_only_installer()
        self.assertEqual(installed.returncode, 0, installed.stdout.decode("utf-8", "replace")
                         + installed.stderr.decode("utf-8", "replace"))

        self.assertTrue(REQUIRED_RUNTIME <= {p.name for p in self.target.iterdir()})
        self.assertEqual(
            (self.target / "answer_phrases.tsv").read_bytes(),
            (PROJECT / "answer_phrases.tsv").read_bytes(),
        )
        self.assertFalse((self.target / ".venv").exists())
        self.assertFalse((self.target / "cache").exists())
        self.assertEqual(hashlib.sha256(db_path.read_bytes()).hexdigest(), before)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
        installed_workflow = self.target / "workflow.py"
        expected_workflow = (PROJECT / "workflow.py").read_bytes()
        installed_workflow.write_bytes(b"# stale sandbox copy\n")
        upgraded = self.run_copy_only_installer()
        self.assertEqual(upgraded.returncode, 0, upgraded.stdout.decode("utf-8", "replace")
                         + upgraded.stderr.decode("utf-8", "replace"))
        self.assertEqual(installed_workflow.read_bytes(), expected_workflow)
        self.assertEqual(hashlib.sha256(db_path.read_bytes()).hexdigest(), before)
        self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
        reopened = storage.open_store(db_path)
        self.assertEqual(reopened.db.execute(
            "SELECT schema_version FROM store_meta").fetchone()[0], storage.SCHEMA_VERSION)
        reopened.close()

    def test_copy_only_refuses_data_outside_marked_sandbox(self):
        data = self.unmarked_dir() / "data"
        data.mkdir()
        self.assert_refused(self.run_copy_only_installer(data=data), self.target)

    def test_copy_only_refuses_marker_that_is_not_a_file(self):
        (self.root / SANDBOX_MARKER).unlink()
        (self.root / SANDBOX_MARKER).mkdir()
        self.assert_refused(self.run_copy_only_installer(), self.target)

    def test_copy_only_refuses_target_with_git_or_venv(self):
        for name in (".git", ".venv"):
            with self.subTest(name=name):
                (self.target / name).mkdir()
                self.assert_refused(self.run_copy_only_installer(), self.target)
                (self.target / name).rmdir()

    def test_copy_only_refuses_junction_inside_sandbox(self):
        if os.name != "nt":
            self.skipTest("Windows junction only")
        outside = self.unmarked_dir()
        link = self.root / "link"
        made = subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(outside)], capture_output=True)
        if made.returncode != 0:
            self.skipTest("creating junction unavailable")
        try:
            target = link / "program"
            self.assert_refused(self.run_copy_only_installer(target=target), outside / "program", outside)
        finally:
            os.rmdir(link)

    def test_previous_schema_client_refuses_v4_without_changing_database(self):
        db_path = self.data / "state.sqlite3"
        store = storage.initialize(db_path)
        store.close()
        before = hashlib.sha256(db_path.read_bytes()).hexdigest()

        older_client = self.root / "older_client"
        older_client.mkdir()
        old_storage = (PROJECT / "storage.py").read_text(encoding="utf-8")
        old_storage = old_storage.replace(
            f"SCHEMA_VERSION = {storage.SCHEMA_VERSION}",
            f"SCHEMA_VERSION = {storage.SCHEMA_VERSION - 1}",
            1,
        )
        (older_client / "storage.py").write_text(old_storage, encoding="utf-8")
        script = (
            "import storage,sys; "
            "storage.open_store(sys.argv[1])"
        )
        result = __import__("subprocess").run(
            [__import__("sys").executable, "-B", "-c", script, str(db_path)],
            cwd=older_client, capture_output=True, text=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("UnsupportedSchema", result.stderr)
        self.assertEqual(hashlib.sha256(db_path.read_bytes()).hexdigest(), before)


class P6ScaleProfileTests(unittest.TestCase):
    def test_synthetic_quote_scale_profile(self):
        tokens = [("слово", "synthetic-speaker", i * 0.2, None) for i in range(10000)]
        tracemalloc.start()
        started = time.perf_counter()
        matches = quotes.candidates(["слово"] * 4, tokens, min_score=1.0)
        elapsed = time.perf_counter() - started
        _, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        self.assertEqual(len(matches), 9997)
        self.assertGreater(peak_bytes, 0)
        # Небольшой loose guard на патологическую регрессию, не обещание скорости.
        self.assertLess(elapsed, 30.0)
        print(json.dumps({
            "profile": "synthetic-quote-candidates-v1",
            "tokens": len(tokens),
            "matches": len(matches),
            "wall_seconds": round(elapsed, 6),
            "peak_tracemalloc_bytes": peak_bytes,
        }, sort_keys=True))

    def test_synthetic_snapshot_storage_profile(self):
        cases = []
        for word_count in (1000, 5000, 10000):
            with tempfile.TemporaryDirectory(prefix="p6-scale-", dir=os.environ["TEMP"]) as temp:
                base = Path(temp)
                root = base / "documents"
                root.mkdir()
                store = storage.initialize(base / "state.sqlite3", initial_display_name="")
                try:
                    identity = store.current_identity()
                    context = store.capture_context(identity["installation_id"], "human")
                    root_id = store.register_root(root)
                    recording_id = str(uuid.uuid4())
                    words = [
                        {"word_id": f"w{i}", "w": "синтетическое", "start": i * 0.4,
                         "end": i * 0.4 + 0.2}
                        for i in range(word_count)
                    ]
                    transcript = {"segments": [{"segment_id": "s1", "speaker": None, "words": words}]}
                    review = {"review": [], "answers": {}}
                    tracemalloc.start()
                    started = time.perf_counter()
                    store.create_recording_with_document(
                        recording_id, "scale", transcript, review, str(uuid.uuid4()), context,
                        root_id=root_id, generated_at="synthetic-profile",
                    )
                    documents.materialize_pending(store, root_id, root)
                    elapsed = time.perf_counter() - started
                    _, peak_bytes = tracemalloc.get_traced_memory()
                    tracemalloc.stop()
                    db_bytes = (base / "state.sqlite3").stat().st_size
                    document_bytes = sum(
                        path.stat().st_size for path in root.rglob("*") if path.is_file()
                    )
                    self.assertGreater(db_bytes, 0)
                    self.assertGreater(document_bytes, 0)
                    self.assertGreater(peak_bytes, 0)
                    self.assertLess(elapsed, 120.0)
                    cases.append({
                        "words": word_count,
                        "wall_seconds": round(elapsed, 6),
                        "database_bytes": db_bytes,
                        "document_bytes": document_bytes,
                        "peak_tracemalloc_bytes": peak_bytes,
                    })
                finally:
                    if tracemalloc.is_tracing():
                        tracemalloc.stop()
                    store.close()
        self.assertEqual([case["words"] for case in cases], [1000, 5000, 10000])
        self.assertEqual(cases, sorted(cases, key=lambda case: case["words"]))
        self.assertLess(cases[0]["document_bytes"], cases[1]["document_bytes"])
        self.assertLess(cases[1]["document_bytes"], cases[2]["document_bytes"])
        print(json.dumps({
            "profile": "synthetic-storage-scale-v1",
            "cases": cases,
        }, sort_keys=True))


class P6InterfaceParityTests(unittest.TestCase):
    def test_workflow_list_matches_across_core_cli_mcp_and_local_screen(self):
        with tempfile.TemporaryDirectory(prefix="p6-parity-", dir=os.environ["TEMP"]) as temp:
            base = Path(temp)
            db_path = base / "state.sqlite3"
            source = base / "synthetic.wav"
            source.write_bytes(b"synthetic-only")
            store = storage.initialize(db_path, initial_display_name="synthetic")
            server = None
            try:
                request_id = workflow.create(store, source)
                with mock.patch.object(paths, "STATE_DB", db_path):
                    expected = core.workflow_list()

                    cli_output = StringIO()
                    args = rt.build().parse_args(["workflow-list"])
                    with redirect_stdout(cli_output):
                        rt.run(args)
                    from_cli = json.loads(cli_output.getvalue())

                    import migrate
                    with mock.patch.object(migrate, "run"):
                        import server as mcp_server
                    from_mcp = json.loads(mcp_server.workflow_list())

                    server, url, _ = workflow_page.make_server(0)
                    thread = threading.Thread(target=server.serve_forever, daemon=True)
                    thread.start()
                    parsed = urlsplit(url)
                    connection = http.client.HTTPConnection(
                        f"127.0.0.1:{server.server_port}", timeout=10)
                    try:
                        connection.request(
                            "GET", parsed.path + "api/requests",
                            headers={"Host": f"127.0.0.1:{server.server_port}"},
                        )
                        response = connection.getresponse()
                        from_screen = json.loads(response.read().decode("utf-8"))
                        connection.request(
                            "GET", parsed.path + "api/status?request_id=00000000-0000-4000-8000-000000000000",
                            headers={"Host": f"127.0.0.1:{server.server_port}"},
                        )
                        error_response = connection.getresponse()
                        error_body = error_response.read().decode("utf-8")
                    finally:
                        connection.close()
                        server.shutdown()
                        thread.join(timeout=5)

                self.assertEqual(from_cli, expected)
                self.assertEqual(from_mcp, expected)
                self.assertEqual(from_screen["requests"], expected)
                self.assertEqual(error_response.status, 400)
                self.assertNotIn("Traceback", error_body)
                self.assertIn("error", json.loads(error_body))
                self.assertEqual([item["request_id"] for item in expected], [request_id])
            finally:
                if server is not None:
                    server.server_close()
                store.close()


if __name__ == "__main__":
    unittest.main()
