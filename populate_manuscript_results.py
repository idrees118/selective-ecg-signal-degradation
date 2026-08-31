#!/usr/bin/env python3
"""Populate manuscript values, tables, and result figures from audited artifacts only.

Run from the PTB-XL project root after the final integrity audit has passed.
This script performs reporting only. It does not train a model, run inference,
fit thresholds, generate corruptions, or recompute bootstrap statistics.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
AUDIT = ROOT / "logs/final_integrity_audit.json"
REPORT = ROOT / "results/reporting/v1/tables"
FINAL = ROOT / "results/final_evaluation/v1/test/condition_metrics.csv"
FIG = ROOT / "figures"


def require_audit_pass() -> None:
    if not AUDIT.exists():
        raise FileNotFoundError(f"Missing {AUDIT}")
    data = json.loads(AUDIT.read_text(encoding="utf-8"))
    if data.get("status") != "PASS":
        raise RuntimeError("Final integrity audit status is not PASS")
    if int(data.get("checks_passed", -1)) != int(data.get("checks_total", -2)):
        raise RuntimeError("Final integrity audit is not a complete pass")


def fmt(value) -> str:
    return f"{float(value):.6f}"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def save_figure(fig, stem: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def generate_values() -> None:
    clean = pd.read_csv(
        REPORT / "clean_test_overall.csv",
        float_precision="round_trip",
    ).iloc[0]

    selective = pd.read_csv(
        REPORT / "selective_aurc_all_conditions.csv",
        float_precision="round_trip",
    )
    clean_selective = selective.loc[
        selective["condition"].eq("clean")
    ].set_index("method")

    values = {
        "CleanTestAUROC": clean["macro_auroc"],
        "CleanTestAP": clean["macro_average_precision"],
        "CleanTestFone": clean["macro_f1"],
        "CleanTestRisk": clean["full_hamming_risk"],
        "CleanTestConfidenceAURC": clean_selective.loc["confidence", "aurc"],
        "CleanTestUAURC": clean_selective.loc["U", "aurc"],
        "CleanTestLUAURC": clean_selective.loc["L+U", "aurc"],
        "CleanTestLQUAURC": clean_selective.loc["L+Q_signal+U", "aurc"],
    }

    lines = ["% Auto-generated from audited reporting artifacts. Do not edit manually."]
    for key, value in values.items():
        lines.append(rf"\newcommand{{\{key}}}{{{fmt(value)}}}")
    (ROOT / "paper_values.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_classwise_table() -> None:
    data = pd.read_csv(
        REPORT / "clean_test_classwise.csv",
        float_precision="round_trip",
    )
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Classwise clean fold-10 test performance.}",
        r"\label{tab:classwise}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Class & AUROC & AP & F1 \\",
        r"\midrule",
    ]
    for _, row in data.iterrows():
        lines.append(
            f"{tex_escape(str(row['class']))} & {fmt(row['auroc'])} & "
            f"{fmt(row['average_precision'])} & {fmt(row['f1'])} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (ROOT / "generated_classwise_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def generate_condition_table(condition_table: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        r"\scriptsize",
        r"\caption{Complete fold-10 performance across clean and corrupted conditions.}",
        r"\label{tab:condition_metrics}",
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        r"Condition & Macro-AUROC & Macro-F1 & Hamming risk & Mean confidence & Mean $Q$ & Mean $U$ \\",
        r"\midrule",
    ]
    for _, row in condition_table.iterrows():
        name = str(row["condition"]).replace("_", " ")
        lines.append(
            f"{name} & {fmt(row['macro_auroc'])} & {fmt(row['macro_f1'])} & "
            f"{fmt(row['full_hamming_risk'])} & {fmt(row['mean_confidence'])} & "
            f"{fmt(row['mean_Q_signal'])} & {fmt(row['mean_U'])} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    (ROOT / "generated_condition_table.tex").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def ordered_series(condition_table: pd.DataFrame, metric: str):
    clean = float(
        condition_table.loc[condition_table["condition"].eq("clean"), metric].iloc[0]
    )
    order = {"mild": 1, "moderate": 2, "severe": 3}
    kinds = (
        ("baseline_wander", "Baseline wander"),
        ("additive_white_noise", "White noise"),
        ("amplitude_clipping", "Amplitude clipping"),
        ("lead_masking", "Lead masking"),
    )
    series = []
    for key, label in kinds:
        group = condition_table.loc[
            condition_table["corruption_type"].eq(key)
        ].copy()
        group["severity_index"] = group["severity"].map(order)
        group = group.sort_values("severity_index")
        if len(group) != 3:
            raise ValueError(f"Expected three severities for {key}")
        y = np.concatenate(([clean], group[metric].to_numpy(dtype=float)))
        series.append((label, y))
    return series


def generate_corruption_figures(condition_table: pd.DataFrame) -> None:
    x = np.arange(4)
    labels = ["Clean", "Mild", "Moderate", "Severe"]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for label, y in ordered_series(condition_table, "full_hamming_risk"):
        ax.plot(x, y, marker="o", label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Full-coverage Hamming risk")
    ax.set_title("Prediction error increases under signal corruption")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "corruption_hamming_risk")

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for label, y in ordered_series(condition_table, "mean_U"):
        ax.plot(x, y, marker="o", label=label)
    ax.set_xticks(x, labels)
    ax.set_xlabel("Corruption severity")
    ax.set_ylabel("Mean MC-dropout uncertainty")
    ax.set_title("Model uncertainty responds to signal degradation")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save_figure(fig, "corruption_mean_uncertainty")


def main() -> None:
    require_audit_pass()
    condition_table = pd.read_csv(FINAL, float_precision="round_trip")
    if condition_table.shape[0] != 13:
        raise ValueError(f"Expected 13 final conditions, found {condition_table.shape[0]}")

    generate_values()
    generate_classwise_table()
    generate_condition_table(condition_table)
    generate_corruption_figures(condition_table)

    print("MANUSCRIPT RESULTS POPULATED FROM AUDITED ARTIFACTS")
    print("Created paper_values.tex")
    print("Created generated_classwise_table.tex")
    print("Created generated_condition_table.tex")
    print("Created figures/corruption_hamming_risk.pdf and .png")
    print("Created figures/corruption_mean_uncertainty.pdf and .png")
    print("No training, inference, tuning, corruption generation, or bootstrap rerun was performed.")


if __name__ == "__main__":
    main()
