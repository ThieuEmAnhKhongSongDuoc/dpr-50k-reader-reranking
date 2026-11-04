"""Plot the saved measurements; no model or GPU execution."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
def read(name):
    return json.loads((ROOT / "results/raw" / name).read_text(encoding="utf-8"))

def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})
    datasets = ["nq", "curatedtrec", "trivia"]
    labels = ["NQ\nn=3,610", "CuratedTREC\nn=694", "TriviaQA\nn=11,313"]
    multi = read("dpr_multi.json")["test_results"]
    base = [100 * multi[k + "_test"]["hits"]["20"] / multi[k + "_test"]["questions"] for k in datasets]
    after = []
    for k in datasets:
        r = read(k + "_reader200.json")["test_results"][k + "_test"]
        after.append(100 * r["hits"]["20"] / r["questions"])
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for bars in [ax.bar(x - .18, base, .36, label="DPR Multi", color="#527a9e"),
                 ax.bar(x + .18, after, .36, label="Reader K=200", color="#dc934a")]:
        ax.bar_label(bars, fmt="%.2f%%", padding=4, fontsize=10)
    ax.set(xticks=x, xticklabels=labels, ylabel="Full-test Hit@20 (%)", ylim=(0, 52),
           title="Reader reranking on the same 50,000-passage corpus")
    ax.legend(loc="upper left", frameon=False)
    ax.yaxis.grid(True, alpha=.2)
    ax.set_axisbelow(True)
    fig.text(.5, .015, "Recorded runs; no confidence intervals estimated.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(ROOT / "images/hit_at_20.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    x = np.arange(2)
    for budget, offset, color in [(100, -.18, "#527a9e"), (200, .18, "#dc934a")]:
        rr = [read(k + "_reader" + str(budget) + ".json")["test_results"][k + "_test"] for k in datasets[:2]]
        axes[0].bar(x + offset, [100 * r["hits"]["20"] / r["questions"] for r in rr], .36, label="K=" + str(budget), color=color)
        axes[1].bar(x + offset, [r["reranking_seconds"] / 60 for r in rr], .36, label="K=" + str(budget), color=color)
    for ax in axes:
        ax.set_xticks(x, ["NQ", "CuratedTREC"])
        ax.legend(frameon=False)
        ax.yaxis.grid(True, alpha=.2)
        ax.set_axisbelow(True)
    axes[0].set(ylabel="Full-test Hit@20 (%)", title="Larger K is not always better")
    axes[1].set(ylabel="Reranking-loop time (minutes)", title="Candidate scoring cost")
    fig.text(.5, .015, "Loop timing excludes setup, model loading, question encoding and FAISS search.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, 1))
    fig.savefig(ROOT / "images/candidate_budget.png", dpi=180)
    plt.close(fig)
    print("Saved images/hit_at_20.png and images/candidate_budget.png")

if __name__ == "__main__":
    main()
