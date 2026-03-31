import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _save_bar_plot(
    out_path: Path,
    title: str,
    x_labels: list[str],
    series: list[tuple[str, list[float]]],
    *,
    ylabel: str,
    ylim=None,
):
    plt.figure(figsize=(10, 5))
    x = list(range(len(x_labels)))
    width = 0.8 / max(1, len(series))

    for si, (sname, values) in enumerate(series):
        offsets = (si - (len(series) - 1) / 2) * width
        plt.bar([xi + offsets for xi in x], values, width=width, label=sname)

    plt.xticks(x, x_labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default="historical_training_run_1")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    li_json = run_dir / "LI" / "li_selection_summary.json"
    lsf_json = run_dir / "LSF" / "lsf_selection_summary.json"
    plots_dir = run_dir / "plots"

    with open(li_json, "r", encoding="utf-8") as f:
        li = json.load(f)
    with open(lsf_json, "r", encoding="utf-8") as f:
        lsf = json.load(f)

    # ----------------
    # LI candidates
    # ----------------
    li_cands = li.get("top_candidates", [])
    # Preserve ordering as stored
    keys = [c["key"] for c in li_cands]

    logloss_vals = [float(c["LogLoss"]) for c in li_cands]
    auroc_vals = [float(c["AUROC"]) for c in li_cands]
    brier_vals = [float(c["Brier"]) for c in li_cands]

    _save_bar_plot(
        plots_dir / "LI_bar_logloss.png",
        title="LI model candidates (Validation/Test Log Loss)",
        x_labels=keys,
        series=[("LogLoss", logloss_vals)],
        ylabel="Log Loss (lower is better)",
    )
    _save_bar_plot(
        plots_dir / "LI_bar_auroc.png",
        title="LI model candidates (Validation/Test AUROC)",
        x_labels=keys,
        series=[("AUROC", auroc_vals)],
        ylabel="AUROC (higher is better)",
        ylim=(0.0, 1.0),
    )
    _save_bar_plot(
        plots_dir / "LI_bar_brier.png",
        title="LI model candidates (Validation/Test Brier Score)",
        x_labels=keys,
        series=[("Brier", brier_vals)],
        ylabel="Brier Score (lower is better)",
    )

    chosen_name = li.get("chosen_name", "")
    # Add a quick comparison plot for chosen vs best candidate
    if len(li_cands) > 0:
        best_idx = min(range(len(li_cands)), key=lambda i: li_cands[i]["LogLoss"])
        best_key = li_cands[best_idx]["key"]
        # Use the chosen test metrics if available from li_selection_summary.json
        # (it includes "test" and "val" blocks)
        test_metrics = li.get("test", {})
        test_logloss = float(test_metrics.get("LogLoss", "nan"))
        test_auroc = float(test_metrics.get("AUROC", "nan"))
        test_brier = float(test_metrics.get("Brier", "nan"))
        # Best candidate values are from top_candidates list; these correspond to those candidate models on the same split used to select (val).
        _save_bar_plot(
            plots_dir / "LI_chosen_vs_best.png",
            title=f"Chosen LI vs Best candidate (chosen={chosen_name})",
            x_labels=["best_candidate", "chosen_model"],
            series=[
                ("LogLoss", [float(li_cands[best_idx]["LogLoss"]), test_logloss]),
                ("AUROC", [float(li_cands[best_idx]["AUROC"]), test_auroc]),
                ("Brier", [float(li_cands[best_idx]["Brier"]), test_brier]),
            ],
            ylabel="Metric value (different scales)",
        )

    # ----------------
    # LSF candidates
    # ----------------
    lsf_cands = lsf.get("candidates", [])
    lsf_keys = [c["key"] for c in lsf_cands]
    rmse_vals = [float(c["RMSE"]) for c in lsf_cands]
    mae_vals = [float(c["MAE"]) for c in lsf_cands]

    _save_bar_plot(
        plots_dir / "LSF_bar_rmse.png",
        title="LSF model candidates (RMSE)",
        x_labels=lsf_keys,
        series=[("RMSE", rmse_vals)],
        ylabel="RMSE (lower is better)",
    )
    _save_bar_plot(
        plots_dir / "LSF_bar_mae.png",
        title="LSF model candidates (MAE)",
        x_labels=lsf_keys,
        series=[("MAE", mae_vals)],
        ylabel="MAE (lower is better)",
    )

    # Optional: chosen vs ensemble_val
    chosen_name_lsf = lsf.get("chosen_name", "")
    test_metrics_lsf = lsf.get("test", {})
    val_metrics_lsf = lsf.get("val", {})
    ensemble_val = lsf.get("ensemble_val", {})
    if ensemble_val:
        _save_bar_plot(
            plots_dir / "LSF_chosen_vs_ensemble_val.png",
            title=f"LSF chosen vs ensemble (chosen={chosen_name_lsf})",
            x_labels=["chosen_on_val", "ensemble_on_val"],
            series=[
                ("RMSE", [float(val_metrics_lsf.get("RMSE", "nan")), float(ensemble_val.get("RMSE", "nan"))]),
                ("MAE", [float(val_metrics_lsf.get("MAE", "nan")), float(ensemble_val.get("MAE", "nan"))]),
            ],
            ylabel="Metric value (different scales)",
        )

    print("Saved plots into:", plots_dir)


if __name__ == "__main__":
    main()

