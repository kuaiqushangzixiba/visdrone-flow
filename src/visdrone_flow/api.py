from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from pydantic import ValidationError

from .model_store import load_bundle
from .schemas import PredictionRequest
from .service import FlowForecastService


def run_server(artifact: str, host: str = "127.0.0.1", port: int = 8010) -> None:
    service = FlowForecastService(load_bundle(artifact))

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/health":
                self._json(200, {"status": "ok", "model": service.bundle.model_name})
                return
            self._json(404, {"error": "not_found"})

        def do_POST(self) -> None:
            if self.path != "/predict":
                self._json(404, {"error": "not_found"})
                return
            try:
                body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
                payload = json.loads(body.decode("utf-8"))
                request = PredictionRequest.model_validate(payload)
                response = service.predict(request)
                self._json(200, response.model_dump(mode="json"))
            except ValidationError as exc:
                self._json(422, {"error": "validation_error", "details": exc.errors()})
            except Exception as exc:  # noqa: BLE001 - API boundary returns structured error.
                self._json(500, {"error": "internal_error", "message": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _json(self, status: int, payload: Any) -> None:
            data = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer((host, port), Handler)
    print(f"visdrone-flow serving on http://{host}:{port}")
    server.serve_forever()

