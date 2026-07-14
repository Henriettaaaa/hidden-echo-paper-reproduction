import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHOD_LABELS = {
    "ldp": "LDP",
    "ldp_h4edu4": "LDP",
    "echo": "HiddenEcho",
    "echo_h4edu4": "HiddenEcho",
    "hiddenecho": "HiddenEcho",
    "echo_plus": "HiddenEcho+",
    "hiddenecho_plus": "HiddenEcho+",
    "no_protection": "No protection",
    "no_protection_h4edu4": "No protection",
    "base_qwen2": "No protection",
}


def infer_method_from_path(path: Path) -> str:
    stem = path.stem
    match = re.search(r"aia_tweet_(.+?)_(education|age)_eta", stem)
    if match:
        return match.group(1)
    return "unknown"


def normalize_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def read_aia_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "method" not in df.columns:
        df["method"] = infer_method_from_path(path)
    expected = {"eta", "method", "attribute", "metric", "value"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    df["source_file"] = str(path)
    df["source_mtime"] = path.stat().st_mtime
    return df


def plot_metric(
    df: pd.DataFrame,
    attribute: str,
    metric: str,
    ylabel: str,
    output: Path,
    figsize: tuple[float, float],
    xticks: list[float],
) -> None:
    sub = df[(df["attribute"] == attribute) & (df["metric"] == metric)].copy()
    if sub.empty:
        print(f"skip {output}: no rows for {attribute}/{metric}")
        return

    sub["eta"] = sub["eta"].astype(float)
    sub["value"] = sub["value"].astype(float)
    sub["method_label"] = sub["method"].map(normalize_method)
    sub = (
        sub.groupby(["eta", "method_label"], as_index=False)["value"]
        .mean()
        .sort_values(["method_label", "eta"])
    )

    budgets = sorted(x for x in sub["eta"].unique() if x > 0)
    if not budgets:
        budgets = [100, 1000, 5000, 6000]
    x_min = 0
    x_max = max(max(budgets), max(xticks))

    plt.figure(figsize=figsize)
    for method, group in sub.groupby("method_label", sort=False):
        group = group.sort_values("eta")
        if method == "No protection":
            y = group["value"].iloc[0]
            plt.plot([x_min, x_max], [y, y], linestyle="--", linewidth=1.5, label=method)
        else:
            group = group[group["eta"] > 0]
            plt.plot(group["eta"], group["value"], marker="o", markersize=3.2, linewidth=1.5, label=method)

    plt.xlabel("Privacy budget eta", fontsize=7)
    plt.ylabel(ylabel, fontsize=7)
    plt.xticks(xticks, [str(int(x)) for x in xticks])
    if metric == "empirical_privacy":
        plt.yticks([0.2, 0.3, 0.4, 0.5, 0.6])
        plt.ylim(0.18, 0.62)
    elif metric == "rmse":
        plt.yticks([11, 12, 13, 14, 15, 16])
        plt.ylim(10.3, 16.2)
    plt.tick_params(axis="both", labelsize=7)
    plt.xlim(x_min, x_max)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(frameon=False, loc="center left", bbox_to_anchor=(1.06, 0.5), fontsize=7)
    plt.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"saved {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, nargs="+", default=[Path("outputs/aia")])
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/aia/figures"))
    parser.add_argument("--fig_width", type=float, default=3.1)
    parser.add_argument("--fig_height", type=float, default=1.5)
    parser.add_argument("--xticks", nargs="+", type=float, default=[5000, 10000])
    args = parser.parse_args()

    csv_paths = []
    for input_dir in args.input_dir:
        csv_paths.extend(sorted(input_dir.glob("*.csv")))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {args.input_dir}")

    df = pd.concat([read_aia_csv(path) for path in csv_paths], ignore_index=True)
    no_protection = df["method"].map(normalize_method) == "No protection"
    if no_protection.any():
        latest_mtime = df.loc[no_protection, "source_mtime"].max()
        df = df[~no_protection | (df["source_mtime"] == latest_mtime)]
    plot_metric(
        df,
        attribute="age",
        metric="rmse",
        ylabel="RMSE",
        output=args.output_dir / "aia_age_rmse.png",
        figsize=(args.fig_width, args.fig_height),
        xticks=args.xticks,
    )
    plot_metric(
        df,
        attribute="education",
        metric="empirical_privacy",
        ylabel="EP",
        output=args.output_dir / "aia_education_ep.png",
        figsize=(args.fig_width, args.fig_height),
        xticks=args.xticks,
    )


if __name__ == "__main__":
    main()
