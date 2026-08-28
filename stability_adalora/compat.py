from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

EXPECTED_PEFT_VERSION = "0.20.0"


def installed_peft_version() -> str | None:
    try:
        return version("peft")
    except PackageNotFoundError:
        return None


def assert_peft_compat() -> str:
    installed = installed_peft_version()
    if installed is None:
        raise RuntimeError(
            "PEFT is not installed. From the project root run: pip install -e ."
        )
    if installed != EXPECTED_PEFT_VERSION:
        raise RuntimeError(
            f"This project targets peft=={EXPECTED_PEFT_VERSION}, but peft=={installed} is installed. "
            "Use a clean Colab runtime and run `pip install -e .` from the project root."
        )
    try:
        from peft.tuners.adalora.layer import RankAllocator 
    except Exception as exc:
        raise RuntimeError(
            "PEFT is installed, but the expected AdaLoRA RankAllocator import is unavailable."
        ) from exc
    return installed
