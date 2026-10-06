#!/usr/bin/env python3
"""Local bridge so the GitHub Pages dashboard can call Queen PAWN-8.

Pages is static and HTTPS. This process must run on the GPU box, and the
public URL you paste into the dashboard must be HTTPS (Cloudflare tunnel
or similar). CPU inference is not supported by the model.

  python bridge.py --model-dir ~/queen_pawn-8 --port 8787
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

MODEL_DIR = Path(os.environ.get("QUEEN_MODEL_DIR", "queen_pawn-8"))
PYTHON = os.environ.get("QUEEN_PYTHON", "")
GPU_MEM = os.environ.get("QUEEN_GPU_MEM", "0.72")
DTYPE = os.environ.get("QUEEN_DTYPE", "bfloat16")


def infer(fen: str, history: list[str] | None) -> dict:
    py = PYTHON or str(MODEL_DIR / ".venv" / "bin" / "python")
    script = MODEL_DIR / "infer.py"
    if not Path(py).exists() or not script.exists():
        raise FileNotFoundError(f"Queen runner not found at {script} / {py}")
    cmd = [
        py, str(script),
        "--fen", fen,
        "--dtype", DTYPE,
        "--gpu-memory-utilization", GPU_MEM,
        "--temperature", "0.6",
        "--max-tokens", "2048",
    ]
    hist_path = None
    if history:
        fd, hist_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        Path(hist_path).write_text(json.dumps(history))
        cmd += ["--history", hist_path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    finally:
        if hist_path:
            Path(hist_path).unlink(missing_ok=True)
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout)[-2000:])
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError("infer.py produced no JSON\n" + proc.stdout[-1500:])
    return json.loads(lines[-1])


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._send(204, {})

    def do_GET(self) -> None:
        if self.path.split("?")[0] in ("/health", "/"):
            ready = (MODEL_DIR / "infer.py").exists()
            self._send(200, {"ok": ready, "model_dir": str(MODEL_DIR), "service": "queen-bridge"})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.split("?")[0] != "/analyze":
            self._send(404, {"error": "not found"})
            return
        n = int(self.headers.get("Content-Length") or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
            fen = str(req["fen"])
            history = req.get("history") or None
            out = infer(fen, history)
            self._send(200, out)
        except Exception as exc:
            self._send(500, {"error": str(exc)})

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args)


def main() -> None:
    global MODEL_DIR, PYTHON, GPU_MEM, DTYPE
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=str(MODEL_DIR))
    p.add_argument("--python", default=PYTHON)
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--gpu-memory-utilization", default=GPU_MEM)
    p.add_argument("--dtype", default=DTYPE)
    args = p.parse_args()
    MODEL_DIR = Path(args.model_dir)
    PYTHON = args.python
    GPU_MEM = args.gpu_memory_utilization
    DTYPE = args.dtype
    httpd = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"queen bridge on :{args.port}  model={MODEL_DIR}")
    httpd.serve_forever()


if __name__ == "__main__":
    main()
