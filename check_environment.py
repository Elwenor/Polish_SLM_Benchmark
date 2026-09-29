#!/usr/bin/env python3

from __future__ import annotations

import importlib.metadata
import json
import sys
from pathlib import Path

import torch
import transformers
from lm_eval import tasks
from lm_eval.models.huggingface import HFLM


EXPECTED_HARNESS_COMMIT = (
    "21d0ea9cf4fd6153dfff4d84d6ad0aab5488f302"
)

EXPECTED_TASKS = [
    "polemo2_in_multiple_choice",
    "polemo2_out_multiple_choice",
    "polish_8tags_multiple_choice",
    "polish_belebele_mc",
    "polish_cbd_multiple_choice",
    "polish_dyk_multiple_choice",
    "polish_klej_ner_multiple_choice",
    "polish_polqa_reranking_multiple_choice",
    "polish_ppc_multiple_choice",
    "polish_psc_multiple_choice",
]


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def get_lm_eval_commit() -> str | None:
    """
    Read VCS metadata written by pip for a direct Git installation.

    Does not print or expose local installation paths.
    """
    try:
        dist = importlib.metadata.distribution("lm_eval")
    except importlib.metadata.PackageNotFoundError:
        return None

    try:
        direct_url = dist.read_text("direct_url.json")
        if not direct_url:
            return None

        data = json.loads(direct_url)

        vcs_info = data.get("vcs_info") or {}
        commit = vcs_info.get("commit_id")

        if commit:
            return str(commit)

    except (json.JSONDecodeError, OSError, TypeError):
        pass

    return None


def fail(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def main() -> None:
    print("=== Polish SLM Benchmark environment check ===\n")

    print("Python:", sys.version.split()[0])
    print("lm_eval:", package_version("lm_eval"))
    print("torch:", torch.__version__)
    print("transformers:", transformers.__version__)
    print("datasets:", package_version("datasets"))
    print("accelerate:", package_version("accelerate"))
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA runtime:", torch.version.cuda or "none")

    print("\n=== lm-evaluation-harness ===")

    commit = get_lm_eval_commit()

    if commit:
        print("Git commit:", commit)

        if commit != EXPECTED_HARNESS_COMMIT:
            print(
                "WARNING: installed lm-evaluation-harness commit differs "
                "from the reference commit."
            )
            print("Expected:", EXPECTED_HARNESS_COMMIT)
    else:
        print(
            "Git commit: unavailable from package metadata "
            "(task compatibility will still be checked)"
        )

    print("\n=== API ===")

    if not hasattr(tasks, "TaskManager"):
        fail(
            "lm_eval.tasks.TaskManager is unavailable.\n"
            "The installed lm-evaluation-harness version is incompatible."
        )

    print("[OK] TaskManager")

    if HFLM is None:
        fail("lm_eval.models.huggingface.HFLM is unavailable.")

    print("[OK] HFLM")

    print("\n=== OpenPL tasks ===")

    try:
        task_manager = tasks.TaskManager()
    except Exception as exc:
        fail(
            "TaskManager could not initialize.\n"
            f"{type(exc).__name__}: {exc}"
        )

    available = set(task_manager.all_tasks)
    missing = []

    for task_name in EXPECTED_TASKS:
        if task_name in available:
            print(f"[OK]      {task_name}")
        else:
            print(f"[MISSING] {task_name}")
            missing.append(task_name)

    if missing:
        print()
        print(
            f"Found {len(EXPECTED_TASKS) - len(missing)}"
            f"/{len(EXPECTED_TASKS)} required OpenPL tasks."
        )

        fail(
            "The installed lm-evaluation-harness does not contain "
            "all tasks required by Polish SLM Benchmark.\n\n"
            "Install the pinned SpeakLeash harness with:\n\n"
            "pip install "
            "\"lm_eval @ "
            "git+https://github.com/speakleash/"
            "lm-evaluation-harness.git@"
            f"{EXPECTED_HARNESS_COMMIT}\""
        )

    print()
    print(
        f"Environment OK: all {len(EXPECTED_TASKS)} "
        "required OpenPL tasks found."
    )


if __name__ == "__main__":
    main()
