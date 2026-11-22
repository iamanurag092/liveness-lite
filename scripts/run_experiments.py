"""Run the quantisation sweep and the ablation trainings back to back (unattended).

    python scripts/run_experiments.py                # everything
    python scripts/run_experiments.py --skip-sweep   # only the trainings
    python scripts/run_experiments.py --only E03_img160

Every run uses 8 epochs on the full cosine schedule without early stopping, because the first
baseline peaked early. A failed step is reported and the next one still runs. Results: runs/E0*/
(history.json, metrics_fp32.json) and the table printed by scripts/summarize_runs.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

CONFIG = "configs/default.yaml"
COMMON = ["train.epochs=8", "train.patience=99"]
EXPERIMENTS = [
    ("E02_base224_8ep", []),
    ("E03_img160", ["data.img_size=160"]),
    ("E04_aug_basic", ["data.augment=basic"]),
    ("E05_mnv3_large", ["model.name=mobilenet_v3_large"]),
]


def run(step: str, command: list[str], status: list[str]) -> bool:
    print(f"\n===== {step} =====\n{' '.join(command)}\n", flush=True)
    started = time.time()
    code = subprocess.run(command, check=False).returncode
    minutes = (time.time() - started) / 60
    status.append(f"{step}: {'ok' if code == 0 else f'FAILED ({code})'} in {minutes:.1f} min")
    return code == 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--only", nargs="*", help="run only these experiment ids")
    args = parser.parse_args()

    py = sys.executable
    status: list[str] = []
    if not args.skip_sweep and not args.only:
        run("quant_sweep", [py, "scripts/quant_sweep.py", "--config", CONFIG], status)

    for run_id, overrides in EXPERIMENTS:
        if args.only and run_id not in args.only:
            continue
        out = f"runs/{run_id}"
        train_cmd = [
            py,
            "-m",
            "liveness.train",
            "--config",
            CONFIG,
            "--override",
            f"output.dir={out}",
            *COMMON,
            *overrides,
        ]
        if run(f"{run_id} train", train_cmd, status):
            eval_cmd = [
                py,
                "-m",
                "liveness.evaluate",
                "--config",
                CONFIG,
                "--checkpoint",
                f"{out}/best.pt",
                "--run-dir",
                out,
                "--tag",
                "fp32",
            ]
            run(f"{run_id} evaluate", eval_cmd, status)

    run("summary", [py, "scripts/summarize_runs.py"], status)
    Path("runs").mkdir(exist_ok=True)
    Path("runs/experiments_status.txt").write_text("\n".join(status) + "\n")
    print("\n".join(["", "===== STATUS ====="] + status))


if __name__ == "__main__":
    main()
