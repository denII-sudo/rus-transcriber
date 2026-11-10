import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import tmpfiles


class TmpFiles(unittest.TestCase):
    def test_scratch_removed_on_drop_and_unknown_prefix_rejected(self):
        p = tmpfiles.scratch("emo_")
        (p / "x.wav").write_bytes(b"1")
        self.assertTrue(p.exists())
        tmpfiles.drop(p)
        self.assertFalse(p.exists())
        with self.assertRaises(ValueError):
            tmpfiles.scratch("evil_")

    def test_removed_at_process_exit(self):
        code = ("import sys; sys.path.insert(0, %r); import tmpfiles; p = tmpfiles.scratch('bench_'); "
                "(p/'a.npy').write_bytes(b'1'); print(p)") % str(Path(__file__).resolve().parent.parent)
        import subprocess
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout.strip()
        self.assertTrue(out)
        self.assertFalse(Path(out).exists(), "временная папка осталась после выхода процесса")

    def test_sweep_stale_only_old_and_only_program_prefixes(self):
        old = Path(tempfile.mkdtemp(prefix="emo_"))
        fresh = Path(tempfile.mkdtemp(prefix="vr_"))
        other = Path(tempfile.mkdtemp(prefix="someone-else-"))
        data = Path(tempfile.mkdtemp(prefix="rt-data-"))
        try:
            past = time.time() - 20 * 3600
            for d in (old, other, data):
                os.utime(d, (past, past))
            tmpfiles.sweep_stale(12)
            self.assertFalse(old.exists())
            self.assertTrue(fresh.exists())
            self.assertTrue(other.exists(), "чужая папка не должна удаляться")
            self.assertTrue(data.exists(), "папки данных (rt-data-) sweep не трогает")
        finally:
            import shutil
            for d in (old, fresh, other, data):
                shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
