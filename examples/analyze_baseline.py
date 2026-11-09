"""Reproduce the lecture tables and figure without a model or API."""

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    frame = pd.read_csv(ROOT / "data/train.csv")
    groups = frame.groupby("Age", dropna=False)["Purchase"].agg(["count", "mean", "median", "sum"])
    print(json.dumps({"rows": len(frame), "columns": len(frame.columns),
                      "unique_users": frame.User_ID.nunique(),
                      "purchase": frame.Purchase.describe().to_dict()}, indent=2))
    print(groups.round(2).to_string())
    if args.plot:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
        fig.set_facecolor("#fbf8f0")
        for ax in axes:
            ax.set_facecolor("#fbf8f0")
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlabel("Age group (source categories)")
            ax.tick_params(axis="x", rotation=25)
        counts = groups["count"] / 1000
        means = groups["mean"]
        axes[0].bar(groups.index, counts, color=["#c88935" if v == counts.max() else "#c4d4cf" for v in counts])
        axes[1].bar(groups.index, means, color=["#177d7b" if v == means.max() else "#c4d4cf" for v in means])
        axes[0].set_title("Most records: 26-35", loc="left", fontweight="bold")
        axes[0].set_ylabel("Purchase records (thousands)")
        axes[1].set_title("Highest mean: 51-55", loc="left", fontweight="bold")
        axes[1].set_ylabel("Mean Purchase (source unit)")
        axes[1].set_ylim(0, 11000)
        for ax, values, decimals in [(axes[0], counts, 1), (axes[1], means, 0)]:
            for i, value in enumerate(values):
                ax.text(i, value, f"{value:,.{decimals}f}", ha="center", va="bottom", fontsize=9)
        fig.suptitle("Same data, different questions", fontsize=19, fontweight="bold", x=.02, ha="left")
        fig.savefig(ROOT / "assets/purchase-by-age.png", dpi=180, facecolor=fig.get_facecolor())
        plt.close(fig)


if __name__ == "__main__":
    main()
