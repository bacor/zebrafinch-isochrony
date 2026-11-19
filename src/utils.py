"""General-purpose helpers for paths and data loading."""

from __future__ import annotations

import os


def get_shared_data_directory() -> str:
    """Path of the shared data directory, set by `SHARED_DATA_DIR` (env or `.env`).

    Raises if the variable is not set or the directory does not exist.
    """
    from dotenv import load_dotenv

    load_dotenv()

    data_dir = os.getenv("SHARED_DATA_DIR")
    if not data_dir:
        raise Exception("SHARED_DATA_DIR environment variable not set")
    if not os.path.exists(data_dir):
        raise Exception(f"Data directory does not exist: {data_dir}")

    return data_dir
