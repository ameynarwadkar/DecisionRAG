import argparse
import socket

import uvicorn

from dec_rag.config import get_settings


def _is_port_available(host: str, port: int) -> bool:
    check_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((check_host, port)) != 0


def _pick_port(host: str, preferred_port: int, explicit: bool) -> int:
    if _is_port_available(host, preferred_port) or explicit:
        return preferred_port
    for candidate in range(preferred_port + 1, preferred_port + 25):
        if _is_port_available(host, candidate):
            print(
                f"Port {preferred_port} is already in use — falling back to port {candidate}."
            )
            return candidate
    return preferred_port


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Run Dec-RAG server & Web Playground")
    parser.add_argument("--host", default=None, help="Bind host (default from HOST / 0.0.0.0)")
    parser.add_argument("--port", type=int, default=None, help="Bind port (default from PORT / 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    host = args.host or settings.HOST
    preferred_port = args.port if args.port is not None else settings.PORT
    port = _pick_port(host, preferred_port, explicit=(args.port is not None))

    display_host = "localhost" if host in {"0.0.0.0", "127.0.0.1"} else host
    print(f"Starting Dec-RAG Playground -> http://{display_host}:{port}/")

    uvicorn.run(
        "dec_rag.main:app",
        host=host,
        port=port,
        reload=args.reload,
    )
