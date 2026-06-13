#!/usr/bin/env python3
"""Minimal Flow123d mock used by the call_flow integration test."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    """Create the files `FlowOutput` expects and echo the input path."""
    input_path = Path(sys.argv[-1])
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "flow123.0.log").write_text(
        "convergence reason 0,\n",
        encoding="utf-8",
    )
    (output_dir / "flow_mock_input.txt").write_text(str(input_path), encoding="utf-8")
    print(f"flow_mock input={input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
