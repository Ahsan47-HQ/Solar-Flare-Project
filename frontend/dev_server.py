"""Standalone local frontend server; the API runs in a different process."""

from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


STATIC = Path(__file__).resolve().parent / "static"


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the Helion frontend")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    args = parser.parse_args()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(STATIC), **handler_kwargs)

        def do_GET(self):
            if self.path == "/config.js":
                content = f"window.HELION_API_BASE = {json.dumps(args.api_base)};\n".encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/javascript; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)
                return
            super().do_GET()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Frontend ready at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
