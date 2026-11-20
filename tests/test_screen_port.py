"""Экран П5 не делит занятый порт с прежним экраном (без моделей и данных)."""
import http.server
import unittest
from urllib.parse import urlsplit

import workflow_page


class BusyPortTests(unittest.TestCase):
    def setUp(self):
        # прежний экран — сервер с настройками по умолчанию (SO_REUSEADDR), как до исправления
        self.old = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        self.addCleanup(self.old.server_close)
        self.busy = self.old.server_address[1]

    def test_busy_port_is_not_shared(self):
        with self.assertRaises(OSError):
            workflow_page.ExclusiveHTTPServer(("127.0.0.1", self.busy), http.server.BaseHTTPRequestHandler)

    def test_screen_takes_next_free_port(self):
        srv, url, _ = workflow_page.make_server(self.busy)
        self.addCleanup(srv.server_close)
        port = srv.server_address[1]
        self.assertNotEqual(port, self.busy)
        self.assertEqual(urlsplit(url).port, port)


if __name__ == "__main__":
    unittest.main()
