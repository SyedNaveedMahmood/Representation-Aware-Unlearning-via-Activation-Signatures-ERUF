#src/llama20/cli.py

import argparse
import importlib
import sys
from typing import Callable, Dict

# Mapping from module short name to (module path, callable name)
MODULES: Dict[str, tuple[str, str]] = {
    "module0": ("llama20.modules.module0", "run_module0"),
    "module_a": ("llama20.modules.module_a", "run_module_a"),
    "module_b": ("llama20.modules.module_b", "run_module_b"),
    "module_c": ("llama20.modules.module_c", "run_module_c"),
    "module_d": ("llama20.modules.module_d", "run_module_d"),
    "module_e": ("llama20.modules.module_e", "run_module_e_tight"),
    "module7": ("llama20.modules.module7", "run_module7_qwen"),
    "module8": ("llama20.modules.module8", "run_module8_clean"),
    "loop": ("llama20.loop", "run_forget_loop"),
}


def load_callable(name: str) -> Callable[[], None]:
    """Import and return the configured callable for a module name."""
    module_path, func_name = MODULES[name]
    module = importlib.import_module(module_path)
    try:
        return getattr(module, func_name)
    except AttributeError as exc:
        raise RuntimeError(f"{module_path} is missing {func_name}") from exc


def run_pipeline(module_names: list[str]) -> None:
    """Run a list of modules sequentially in the given order."""
    for name in module_names:
        if name not in MODULES:
            raise ValueError(f"Unknown module '{name}'. Valid options: {sorted(MODULES)}")
        print(f"==> Running {name}")
        func = load_callable(name)
        func()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Dispatches into the notebook modules extracted into llama20.*",
    )
    parser.add_argument(
        "module",
        choices=sorted(list(MODULES.keys()) + ["pipeline"]),
        help="Module to run, or 'pipeline' to run multiple in sequence.",
    )
    parser.add_argument(
        "--modules",
        nargs="+",
        default=[
            "module0",
            "module_a",
            "module_b",
            "module_c",
            "module_d",
            "module_e",
            "module7",
            "module8",
        ],
        help="When module=pipeline, run these modules in order.",
    )
    args = parser.parse_args(argv)

    if args.module == "pipeline":
        run_pipeline(args.modules)
        return

    func = load_callable(args.module)
    func()


if __name__ == "__main__":
    main(sys.argv[1:])
