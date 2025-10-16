"""Legacy CLI stub retained for backward compatibility."""

from __future__ import annotations

import sys

MSG = (
    "\nThe monolithic CLI has been replaced by:\n"
    "  - python -m learn_noise.cli_2d [OPTIONS]\n"
    "  - python -m learn_noise.cli_images [OPTIONS]\n\n"
    "Please migrate to the new entry points.\n"
)


def main() -> None:
    sys.stderr.write(MSG)
    sys.exit(1)


if __name__ == "__main__":
    main()
