"""
Entry point for the ASR API server.

Usage:
    python serve.py
    python serve.py --host 0.0.0.0 --port 8000 --device cuda
"""

import argparse
import logging
import uvicorn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)


def main():
    parser = argparse.ArgumentParser(description="Unified ASR API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--workers", type=int, default=2, help="Thread pool workers")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    from server.config import ServerConfig
    import server.api as api_module

    config = ServerConfig(
        host=args.host,
        port=args.port,
        device=args.device,
        max_workers=args.workers,
        debug=args.debug,
    )

    from contextlib import asynccontextmanager
    from concurrent.futures import ThreadPoolExecutor

    @asynccontextmanager
    async def patched_lifespan(app):
        from server.engine import ASREngine

        api_module._config = config
        logging.getLogger("asr.api").info("Starting ASR server...")

        api_module._engine = ASREngine(config)
        api_module._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        logging.getLogger("asr.api").info(
            "ASR server ready on %s:%d", config.host, config.port
        )
        yield

        api_module._executor.shutdown(wait=False)
        logging.getLogger("asr.api").info("ASR server shut down")

    api_module.app.router.lifespan_context = patched_lifespan

    uvicorn.run(
        api_module.app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
    )


if __name__ == "__main__":
    main()
