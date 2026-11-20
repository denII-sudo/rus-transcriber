"""Страница карточек не делит занятый порт с прежней страницей (без моделей и данных)."""
import http.server
import unittest

import review_page


class BusyPortTests(unittest.TestCase):
    def setUp(self):
        # прежняя страница — сервер с настройками по умолчанию (SO_REUSEADDR), как до исправления
        self.old = http.server.ThreadingHTTPServer(("127.0.0.1", 0), http.server.BaseHTTPRequestHandler)
        self.addCleanup(self.old.server_close)
        self.busy = self.old.server_address[1]

    def test_busy_port_is_not_shared(self):
        with self.assertRaises(OSError):
            review_page.ExclusiveHTTPServer(("127.0.0.1", self.busy), http.server.BaseHTTPRequestHandler)

    def test_page_takes_next_free_port(self):
        srv, port = review_page.bind_free_port(self.busy, http.server.BaseHTTPRequestHandler)
        self.assertIsNotNone(srv)
        self.addCleanup(srv.server_close)
        self.assertNotEqual(port, self.busy)
        self.assertEqual(srv.server_address[1], port)


if __name__ == "__main__":
    unittest.main()
