"""Build the results figures from ``results/*.csv``.

    uv run python experiments/make_figures.py
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import FIGDIR, LAMBDA_REF, RESULTS

mpl.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
})

PALETTE = {
    "Oracle": "#1a202c",
    "Supervised(ridge)": "#2f855a",
    "LinUCB(a=0.3)": "#2b6cb0",
    "D-LinUCB(a=0.3,g=0.9995)": "#805ad5",
    "LinTS(v=0.2)": "#dd6b20",
    "eps-greedy(e=0.1)": "#d53f8c",
    "ZeroRouter": "#718096",
    "Random": "#a0aec0",
    "always-cheapest": "#b7791f",
    "always-premium": "#c53030",
}


def colour(name: str) -> str:
    for key, value in PALETTE.items():
        if name.startswith(key.split("(")[0]):
            return value
    return "#4a5568"


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}")
    print(f"saved -> results/figures/{name}.pdf")


# --------------------------------------------------------------------------- #
def figure_rq1() -> None:
    df = pd.read_csv(RESULTS / "rq1.csv")
    agg = df.groupby(["policy", "lam"]).agg(
        reward=("mean_reward", "mean"),
        quality=("mean_quality", "mean"),
        cost=("mean_dollar_cost", "mean"),
        regret=("total_regret", "mean"),
        regret_sd=("total_regret", "std"),
    ).reset_index()

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 3.9))

    # (a) achieved cost-quality frontier, swept over lambda
    for policy, grp in agg.groupby("policy"):
        grp = grp.sort_values("cost")
        style = "--" if policy in ("Oracle", "Random") else "-"
        axL.plot(grp["cost"], grp["quality"], style, marker="o", ms=3.4, lw=1.5,
                 color=colour(policy), label=policy, alpha=0.9)
    axL.set_xscale("log")
    axL.set_xlabel("mean cost per query (USD, log scale)")
    axL.set_ylabel("mean quality")
    axL.set_title(r"(a) Achieved cost-quality frontier (sweeping $\lambda$)")
    axL.legend(frameon=False, ncol=2, fontsize=7, loc="lower right")

    # (b) total regret at the reference lambda, with bootstrap-free SD bars
    ref = agg[np.isclose(agg["lam"], LAMBDA_REF)].sort_values("regret")
    ref = ref[ref["policy"] != "Oracle"]
    ypos = np.arange(len(ref))
    axR.barh(ypos, ref["regret"], xerr=ref["regret_sd"].fillna(0.0),
             color=[colour(p) for p in ref["policy"]], alpha=0.9,
             error_kw={"lw": 0.9, "capsize": 2.5})
    axR.set_yticks(ypos)
    axR.set_yticklabels(ref["policy"], fontsize=7.5)
    axR.invert_yaxis()
    axR.set_xlabel("cumulative regret vs per-query oracle (lower is better)")
    axR.set_title(rf"(b) Regret at $\lambda={LAMBDA_REF:g}$ (mean $\pm$ sd over seeds)")
    axR.grid(axis="y", alpha=0)

    fig.tight_layout()
    save(fig, "fig4_rq1_frontier_regret")
    plt.close(fig)


def figure_rq2() -> None:
    df = pd.read_csv(RESULTS / "rq2.csv")
    scenarios = [s for s in df["scenario"].unique() if s != "S0-stationary"]
    base = df[df["scenario"] == "S0-stationary"].groupby("policy")["post_reward"].mean()

    agg = df.groupby(["scenario", "policy"]).agg(
        post=("post_reward", "mean"), sd=("post_reward", "std")
    ).reset_index()
    agg["drop"] = agg.apply(lambda r: base.get(r["policy"], np.nan) - r["post"], axis=1)

    policies = sorted(df["policy"].unique())
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    width = 0.8 / len(policies)
    xs = np.arange(len(scenarios))
    for i, policy in enumerate(policies):
        sub = agg[agg["policy"] == policy].set_index("scenario").reindex(scenarios)
        ax.bar(xs + i * width, sub["drop"], width, yerr=sub["sd"].fillna(0.0),
               label=policy, color=colour(policy), alpha=0.9,
               error_kw={"lw": 0.8, "capsize": 2})
    ax.axhline(0, color="#1a202c", lw=0.8)
    ax.set_xticks(xs + width * (len(policies) - 1) / 2)
    ax.set_xticklabels(scenarios, fontsize=8)
    ax.set_ylabel("post-shift reward lost\nvs stationary (lower is better)")
    ax.set_title("Degradation under each scripted shift")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    fig.tight_layout()
    save(fig, "fig5_rq2_shifts")
    plt.close(fig)


def figure_rq3() -> None:
    df = pd.read_csv(RESULTS / "rq3.csv")
    agg = df.groupby(["logger", "estimator"]).agg(
        bias=("bias", "mean"),
        rmse=("abs_error", lambda v: float(np.sqrt(np.mean(np.square(v))))),
        minp=("min_propensity", "mean"),
    ).reset_index()

    loggers = list(dict.fromkeys(df["logger"]))
    estimators = ["IPS", "SNIPS", "DR"]
    colours = {"IPS": "#c53030", "SNIPS": "#dd6b20", "DR": "#2f855a"}

    fig, (a0, a1) = plt.subplots(1, 2, figsize=(10.0, 3.5))
    xs = np.arange(len(loggers))
    width = 0.26
    for i, est in enumerate(estimators):
        sub = agg[agg["estimator"] == est].set_index("logger").reindex(loggers)
        a0.bar(xs + i * width, sub["bias"], width, label=est, color=colours[est], alpha=0.9)
        a1.bar(xs + i * width, sub["rmse"], width, label=est, color=colours[est], alpha=0.9)
    for ax, title, ylab in (
        (a0, "(a) Estimator bias vs simulator ground truth", "mean bias"),
        (a1, "(b) RMSE", "RMSE"),
    ):
        ax.set_xticks(xs + width)
        ax.set_xticklabels(loggers, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.legend(frameon=False, fontsize=8)
    a0.axhline(0, color="#1a202c", lw=0.8)
    fig.tight_layout()
    save(fig, "fig6_rq3_ope")
    plt.close(fig)


def main() -> None:
    for name, fn in (("rq1", figure_rq1), ("rq2", figure_rq2), ("rq3", figure_rq3)):
        path = RESULTS / f"{name}.csv"
        if path.exists():
            fn()
        else:
            print(f"skipping {name}: {path} not found")


if __name__ == "__main__":
    main()
