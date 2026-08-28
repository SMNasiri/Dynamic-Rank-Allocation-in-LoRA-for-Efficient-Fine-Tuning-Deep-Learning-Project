from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

from stability_adalora.compat import EXPECTED_PEFT_VERSION, assert_peft_compat


def pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    peft_version = assert_peft_compat()
    print("Environment check: OK")
    print(f"Python:       {platform.python_version()}")
    print(f"PEFT:         {peft_version} (expected {EXPECTED_PEFT_VERSION})")
    print(f"Transformers: {pkg('transformers')}")
    print(f"Torch:        {pkg('torch')}")
    print(f"Datasets:     {pkg('datasets')}")
    print("AdaLoRA integration point: peft.tuners.adalora.layer.RankAllocator")


if __name__ == "__main__":
    main()
