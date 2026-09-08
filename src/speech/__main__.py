"""Start the speech process."""

from __future__ import annotations

import logging

import uvicorn

from speech.config import load_settings
from speech.cuda_libs import prepare_cuda_libraries
from speech.service import create_app


def main() -> None:
    """Read the environment, hold both models, and serve the contract."""
    prepare_cuda_libraries()
    settings = load_settings()
    logging.basicConfig(
        level=logging.DEBUG if settings.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    app = create_app(settings)
    uvicorn.run(app, host=settings.host, port=settings.port)


if __name__ == "__main__":
    main()
