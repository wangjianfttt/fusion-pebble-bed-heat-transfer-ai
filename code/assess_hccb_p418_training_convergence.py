#!/usr/bin/env python3
"""Check whether the first P418 neural-model comparison stops while still improving."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


ARCHITECTURE_SOURCE_NAMES = {
    "pinn_data_only": "PINO-paper coordinate PINN control",
    "pinn": "PINO-paper coordinate PINN control",
    "graph": "RIGNO-style regional graph operator",
    "transolver": "Transolver",
}


def published_epochs(registry: Path) -> dict[str, int]:
    payload = json.loads(registry.read_text(encoding="utf-8"))
    records = {
        str(record["name"]): record
        for record in payload.get("architectures", [])
        if isinstance(record, dict) and "name" in record
    }
    result: dict[str, int] = {}
    for architecture, source_name in ARCHITECTURE_SOURCE_NAMES.items():
        if source_name not in records:
            raise ValueError(f"architecture source is absent from registry: {source_name}")
        settings = records[source_name].get("source_settings", {})
        epochs = settings.get("epochs") if isinstance(settings, dict) else None
        if epochs is None or int(epochs) <= 0:
            raise ValueError(f"published epoch count is absent for {source_name}")
        result[architecture] = int(epochs)
    return result


def load_validation_history(path: Path, expected_epochs: int) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"missing training history: {path}")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    epochs: list[int] = []
    losses: list[float] = []
    for record in records:
        epoch = int(record["epoch"])
        validation = record.get("validation")
        if not isinstance(validation, dict) or "total_loss" not in validation:
            raise ValueError(f"validation total_loss is absent at epoch {epoch}: {path}")
        loss = float(validation["total_loss"])
        if not math.isfinite(loss) or loss <= 0.0:
            raise ValueError(f"validation total_loss must be finite and positive at epoch {epoch}: {path}")
        epochs.append(epoch)
        losses.append(loss)
    expected = list(range(1, expected_epochs + 1))
    if epochs != expected:
        raise ValueError(
            f"training history is incomplete or non-consecutive: {path}; "
            f"found {epochs[:3]}...{epochs[-3:] if epochs else []}, expected 1...{expected_epochs}"
        )
    return np.asarray(epochs, dtype=float), np.asarray(losses, dtype=float)


def assess_history(
    epochs: np.ndarray,
    losses: np.ndarray,
    *,
    source_epochs: int,
) -> dict[str, object]:
    completed = int(epochs[-1])
    best_index = int(np.argmin(losses))
    best_epoch = int(epochs[best_index])
    best_at_training_boundary = best_epoch == completed
    if best_at_training_boundary:
        status = "training_extension_required_before_final_accuracy_claim"
        interpretation_cn = (
            "最佳验证误差仍出现在训练末段，当前轮数只能作为第一轮比较，"
            "需要延长训练后再报告最终精度。"
        )
    else:
        status = "best_validation_checkpoint_before_training_boundary"
        interpretation_cn = (
            "最佳验证误差没有卡在训练终点，可使用该最佳检查点进行本轮比较；"
            "这不等同于已经证明达到全局最优。"
        )
    return {
        "completed_epochs": completed,
        "published_source_epochs": source_epochs,
        "completed_fraction_of_published_schedule": completed / source_epochs,
        "best_epoch": best_epoch,
        "best_validation_total_loss": float(losses[best_index]),
        "final_validation_total_loss": float(losses[-1]),
        "final_to_best_loss_ratio": float(losses[-1] / losses[best_index]),
        "best_epoch_is_final_epoch": best_at_training_boundary,
        "status": status,
        "interpretation_cn": interpretation_cn,
    }


def write_chinese_summary(path: Path, rows: list[dict[str, object]], requested_epochs: int) -> None:
    lines = [
        "# P418 四模型训练稳定性判断",
        "",
        f"本文件检查 {requested_epochs} 轮训练结束时，验证误差是否仍在继续改善。",
        "这里不设置末段百分比或容差：只检查最低验证误差是否恰好出现在最后一轮。",
        "",
        "| 模型 | 数据划分 | 最佳轮次 | 原论文轮数 | 当前占比 | 判断 |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            "| {architecture} | {split} | {best_epoch}/{completed_epochs} | "
            "{published_source_epochs} | {fraction:.1%} | {interpretation_cn} |".format(
                **row,
                fraction=float(row["completed_fraction_of_published_schedule"]),
            )
        )
    lines.extend(
        [
            "",
            "## 使用原则",
            "",
            "- 如果最低验证误差恰好位于最后一轮，当前训练长度不足以看到验证误差回升或稳定，按该方法公开来源中的轮数从头重训。",
            "- 如果最佳检查点位于前面，可以用最佳检查点做本轮比较，但仍需结合训练曲线和独立测试工况判断。",
            "- 程序在JSON中给出原论文周期对照：需要延长时从头运行该方法原论文轮数，否则保留本轮训练长度。长周期结果不预设更好，必须与短周期结果逐项比较工程量。",
            "- PINN与纯数据PINN使用相同网络和训练设置，因此两者的差别来自质量守恒和能量守恒约束。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--result-prefix", default="hccb_p418_60")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument(
        "--architectures",
        nargs="+",
        choices=tuple(ARCHITECTURE_SOURCE_NAMES),
        default=list(ARCHITECTURE_SOURCE_NAMES),
    )
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--architecture-registry", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    source_epoch_map = published_epochs(args.architecture_registry)
    rows: list[dict[str, object]] = []
    for split in args.splits:
        for architecture in args.architectures:
            directory = (
                args.results_root
                / f"{args.result_prefix}_{architecture}_{split}_{args.epochs}epoch"
            )
            history = directory / "training_history.jsonl"
            epochs, losses = load_validation_history(history, args.epochs)
            row = {
                "architecture": architecture,
                "split": split,
                "history_file": str(history.resolve()),
                **assess_history(
                    epochs,
                    losses,
                    source_epochs=source_epoch_map[architecture],
                ),
            }
            rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "training_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "status": "complete",
        "requested_epochs": args.epochs,
        "result_prefix": args.result_prefix,
        "assessment_rule": (
            "A best validation checkpoint exactly at the final completed epoch is a "
            "right-boundary warning. No fractional window or tolerance is introduced."
        ),
        "training_extension_required_count": sum(
            bool(row["best_epoch_is_final_epoch"]) for row in rows
        ),
        "recommended_followup_epochs": {
            str(split): {
                str(row["architecture"]): (
                    int(row["published_source_epochs"])
                    if bool(row["best_epoch_is_final_epoch"])
                    else int(row["completed_epochs"])
                )
                for row in rows
                if row["split"] == split
            }
            for split in dict.fromkeys(str(row["split"]) for row in rows)
        },
        "recommended_followup_rule": (
            "If the best validation checkpoint is exactly the final epoch of the first "
            "comparison, rerun from scratch with the epoch count reported by that method's "
            "archived source. Retain both runs and compare the engineering quantities separately; "
            "the longer source schedule is a comparison, not a prediction of better performance. "
            "Otherwise retain the completed validation-selected run."
        ),
        "models": rows,
        "new_physical_parameters": [],
    }
    (args.output_dir / "training_convergence.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_chinese_summary(
        args.output_dir / "TRAINING_CONVERGENCE_CN.md",
        rows,
        args.epochs,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
