"""
Helpers for retrieving sample file paths from environment variables.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

load_dotenv()

SINGLE_CHECK_SAMPLE_ENV = "SINGLE_CHECK_SAMPLE_PDF"
MULTI_CHECK_SAMPLE_ENV = "MULTI_CHECK_SAMPLE_PDF"
ALTERNATE_SINGLE_CHECK_SAMPLE_ENV = "ALTERNATE_SINGLE_CHECK_SAMPLE_PDF"
DEMO_CHECK_SAMPLE_ENV = "DEMO_CHECK_SAMPLE_PDF"
CHECK_DETECTION_SAMPLE_LIST_ENV = "CHECK_DETECTION_SAMPLE_PDFS"

def _expand_and_validate(path_value: str, description: str) -> Path:
    """Expand user/home references and confirm the path exists."""
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"{description} not found at {path}")
    return path


def _require_env_value(env_var: str, description: str) -> str:
    """Ensure an environment variable is provided."""
    value = os.environ.get(env_var)
    if not value:
        raise RuntimeError(f"Set {env_var} to the {description}")
    return value


def get_sample_pdf_path(env_var: str, description: str) -> Path:
    """Resolve a sample PDF path using a dedicated environment variable."""
    value = _require_env_value(env_var, description)
    return _expand_and_validate(value, description)


def get_single_check_sample_pdf() -> Path:
    """Return the configured single-check PDF sample path."""
    return get_sample_pdf_path(
        SINGLE_CHECK_SAMPLE_ENV, "path of the single-check PDF sample"
    )


def get_multi_check_sample_pdf() -> Path:
    """Return the configured multi-check PDF sample path."""
    return get_sample_pdf_path(
        MULTI_CHECK_SAMPLE_ENV, "path of the multi-check PDF sample"
    )


def get_alternate_single_check_pdf() -> Path:
    """Return the configured alternate single-check sample path."""
    return get_sample_pdf_path(
        ALTERNATE_SINGLE_CHECK_SAMPLE_ENV,
        "path of the alternate single-check PDF sample",
    )


def get_demo_check_sample_pdf() -> Path:
    """Return the configured demo/check attachment PDF sample path."""
    return get_sample_pdf_path(
        DEMO_CHECK_SAMPLE_ENV, "path of the demo check PDF used for attachments"
    )


def get_sample_pdf_list(env_var: str, description: str) -> List[Path]:
    """Parse a comma separated list of sample PDF paths."""
    value = _require_env_value(env_var, description)
    paths = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        paths.append(_expand_and_validate(item, description))
    if not paths:
        raise RuntimeError(f"{env_var} did not contain any valid paths")
    return paths
