"""Emit LaTeX tables for the report from ``results/*.csv``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import LAMBDA_REF, REPO, RESULTS, is_frontier

from router_bandit.evaluation import (
    average_performance_gap_recovered,
    cost_performance_threshold,
)

TABDIR = REPO / "report" / "tables"
TABDIR.mkdir(parents=True, exist_ok=True)

SHORT = {
    "Oracle": "Per-query oracle",
    "Supervised(ridge)": "Supervised (full info)",
    "LinUCB(a=0.3)": "LinUCB",
    "LinTS(v=0.2)": "LinTS",
    "D-LinUCB(a=0.3,g=0.9995)": "Discounted LinUCB",
    "PILOT(a=0.3)": "PILOT-style",
    "eps-greedy(e=0.1)": "$\\epsilon$-greedy",
    "always-cheapest": "Always cheapest",
    "always-premium": "Always premium",
    "Random": "Random",
    "Cascade(e=0)": "Cascade (perfect judge)",
    "Cascade(e=0.1)": "Cascade (10\\% judge error)",
    "Cascade(e=0.2)": "Cascade (20\\% judge error)",
}


def pretty(name: str) -> str:
    if name.startswith("ZeroRouter"):
        return "Zero Router"
    return SHORT.get(name, name)


def write(text: str, name: str) -> None:
    (TABDIR / f"{name}.tex").write_text(text)
    print(f"saved -> report/tables/{name}.tex")


def _blank(value: float, spec: str) -> str:
    """Render a missing recovery or threshold as an empty-looking dash."""
    if pd.isna(value):
        return "--"
    return format(value, spec)


def _ci(lo: float, hi: float, spec: str = ".4f") -> str:
    if pd.isna(lo) or pd.isna(hi):
        return "--"
    return f"$[{format(lo, spec)},{format(hi, spec)}]$"


def table_rq1() -> None:
    df = pd.read_csv(RESULTS / "rq1.csv")
    # Hull-sweep rows are the frontier curve, drawn in Figure 4a. The tables
    # keep the single lambda-matched Zero Router.
    df = df.loc[~is_frontier(df)].copy()
    ref = df[np.isclose(df["lam"], LAMBDA_REF)]
    if "heldout_reward" not in ref.columns:
        _table_rq1_stream(ref)
        print("rq1.csv has no held-out columns; wrote the stream-only table")
        return
    _table_rq1_heldout(ref)
    _table_rq1_latency(ref)
    _table_apgr(df)


def _table_rq1_stream(ref: pd.DataFrame) -> None:
    """Stream means, for a results file written before frozen held-out eval."""
    agg = ref.groupby("policy").agg(
        reward=("mean_reward", "mean"), reward_sd=("mean_reward", "std"),
        quality=("mean_quality", "mean"), cost=("mean_dollar_cost", "mean"),
        regret=("total_regret", "mean"), match=("oracle_match", "mean"),
    ).reset_index().sort_values("reward", ascending=False)

    sup = float(agg.loc[agg["policy"] == "Supervised(ridge)", "reward"].iloc[0])
    lines = [
        r"\begin{tabular}{lrrrrrr}", r"\toprule",
        r"Policy & Reward & \% Sup. & Quality & Cost (\$) & Regret & Oracle \% \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        sd = 0.0 if np.isnan(r["reward_sd"]) else r["reward_sd"]
        lines.append(
            f"{pretty(r['policy'])} & ${r['reward']:.4f}\\pm{sd:.4f}$ & "
            f"{100 * r['reward'] / sup:.1f} & {r['quality']:.4f} & "
            f"{r['cost']:.2e} & {r['regret']:.0f} & {100 * r['match']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_rq1")


def _table_rq1_heldout(ref: pd.DataFrame) -> None:
    """Frozen held-out means at the reference lambda.

    Seed-to-seed standard deviation is its own column. Each interval is the
    mean of the per-seed bootstrap bounds, which were computed over queries.
    """
    agg = ref.groupby("policy").agg(
        reward=("heldout_reward", "mean"), reward_sd=("heldout_reward", "std"),
        reward_lo=("heldout_reward_lo", "mean"), reward_hi=("heldout_reward_hi", "mean"),
        quality=("heldout_quality", "mean"),
        quality_lo=("heldout_quality_lo", "mean"), quality_hi=("heldout_quality_hi", "mean"),
        cost=("heldout_dollar_cost", "mean"),
        cost_lo=("heldout_dollar_cost_lo", "mean"), cost_hi=("heldout_dollar_cost_hi", "mean"),
        regret=("total_regret", "mean"), match=("oracle_match", "mean"),
    ).reset_index().sort_values("reward", ascending=False)

    sup = float(agg.loc[agg["policy"] == "Supervised(ridge)", "reward"].iloc[0])
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}", r"\toprule",
        (
            r"Policy & Reward & Seed SD & 95\% CI & Quality & 95\% CI & "
            r"Cost (\$) & 95\% CI & \% Sup. & Regret & Oracle \% \\"
        ),
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        sd = 0.0 if np.isnan(r["reward_sd"]) else r["reward_sd"]
        lines.append(
            f"{pretty(r['policy'])} & {r['reward']:.4f} & {sd:.4f} & "
            f"{_ci(r['reward_lo'], r['reward_hi'])} & {r['quality']:.4f} & "
            f"{_ci(r['quality_lo'], r['quality_hi'])} & {r['cost']:.2e} & "
            f"{_ci(r['cost_lo'], r['cost_hi'], '.2e')} & {100 * r['reward'] / sup:.1f} & "
            f"{r['regret']:.0f} & {100 * r['match']:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_rq1")


def _table_rq1_latency(ref: pd.DataFrame) -> None:
    agg = ref.groupby("policy").agg(
        p50=("select_p50_s", "mean"), p99=("select_p99_s", "mean"),
    ).reset_index().sort_values("p50")
    # The cascade is scored off the single-pull loop, so it has no select time.
    agg = agg.replace([np.inf, -np.inf], np.nan).dropna(subset=["p50"])
    lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Policy & Select p50 (ms) & Select p99 (ms) \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{pretty(r['policy'])} & {1e3 * r['p50']:.3f} & {1e3 * r['p99']:.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_rq1_latency")


def _table_apgr(df: pd.DataFrame) -> None:
    """APGR over the lambda grid, and the cheapest frontier point that recovers half the gap."""
    by_lam = df.groupby(["policy", "lam"], as_index=False).agg(
        pgr=("pgr", "mean"), cost=("heldout_dollar_cost", "mean"),
    )
    rows = []
    for policy, grp in by_lam.groupby("policy"):
        pgr = grp["pgr"].to_numpy()
        cost = grp["cost"].to_numpy()
        rows.append({
            "policy": policy,
            "apgr": average_performance_gap_recovered(pgr),
            "cpt": cost_performance_threshold(cost, pgr),
        })
    agg = pd.DataFrame(rows).sort_values("apgr", ascending=False)
    lines = [
        r"\begin{tabular}{lrr}", r"\toprule",
        r"Policy & APGR & CPT (\$) \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{pretty(r['policy'])} & {_blank(r['apgr'], '.3f')} & {_blank(r['cpt'], '.2e')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_apgr")


def table_rq2() -> None:
    df = pd.read_csv(RESULTS / "rq2.csv")
    base = df[df["scenario"] == "S0-stationary"].groupby("policy")["post_reward"].mean()
    piv = df.pivot_table(index="policy", columns="scenario", values="post_reward", aggfunc="mean")
    scen = [c for c in piv.columns if c != "S0-stationary"]
    loss = piv[scen].rsub(base, axis=0)

    lines = [
        r"\begin{tabular}{l" + "r" * (len(scen) + 1) + "}", r"\toprule",
        "Policy & Stationary & " + " & ".join(s.split("-", 1)[0] for s in scen) + r" \\",
        r"\midrule",
    ]
    for policy in piv.index:
        cells = " & ".join(f"{loss.loc[policy, s]:+.4f}" for s in scen)
        lines.append(f"{pretty(policy)} & {base[policy]:.4f} & {cells} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
    ]
    write("\n".join(lines), "res_rq2")
    _table_rq2_recovery(df)


def _table_rq2_recovery(df: pd.DataFrame) -> None:
    """Post-shift regret is the headline; departure and return are blank when absent."""
    if "departed_at" not in df.columns:
        print("rq2.csv has no recovery columns; skipped the recovery table")
        return
    agg = df.groupby(["policy", "scenario"], as_index=False).agg(
        regret=("post_regret", "mean"),
        departed=("departed_at", "mean"),
        recover=("rounds_to_recover", "mean"),
    )
    lines = [
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Policy & Scenario & Post-shift regret & Departed & Recovered \\",
        r"\midrule",
    ]
    for _, r in agg.iterrows():
        lines.append(
            f"{pretty(r['policy'])} & {r['scenario']} & {r['regret']:.1f} & "
            f"{_blank(r['departed'], '.0f')} & {_blank(r['recover'], '.0f')} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_rq2_recovery")


def table_rq3() -> None:
    df = pd.read_csv(RESULTS / "rq3.csv")
    agg = df.groupby(["logger", "estimator"]).agg(
        bias=("bias", "mean"),
        rmse=("abs_error", lambda v: float(np.sqrt(np.mean(np.square(v))))),
        minp=("min_propensity", "mean"),
    ).reset_index()

    loggers = list(dict.fromkeys(df["logger"]))
    lines = [
        r"\begin{tabular}{llrrr}", r"\toprule",
        r"Logging policy & Est. & Bias & RMSE & $\min p$ \\", r"\midrule",
    ]
    for lg in loggers:
        sub = agg[agg["logger"] == lg]
        for i, (_, r) in enumerate(sub.iterrows()):
            label = lg.replace("_", r"\_") if i == 0 else ""
            lines.append(
                f"{label} & {r['estimator']} & {r['bias']:+.4f} & "
                f"{r['rmse']:.4f} & {r['minp']:.3f} \\\\"
            )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_rq3")


def table_pool() -> None:
    """Model-pool table, LaTeX-safe (the notebook version leaves $ unescaped)."""
    import sys
    sys.path.insert(0, str(REPO / "src"))
    from router_bandit.data import load, short_names
    full = sorted(REPO.glob("data/raw/**/routerbench_0shot.pkl"))
    src = full[0] if full else REPO / "data" / "fixtures" / "rb0_sample5k.parquet"
    d = load(src)
    Q, C = d.quality, d.cost
    o = d.oracle(LAMBDA_REF)
    share = np.bincount(o, minlength=d.n_arms) / d.n_queries
    names = short_names()
    order = np.argsort(C.mean(0))

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Model & Quality & Cost (USD) & $\times$cheap & Oracle \% \\", r"\midrule"]
    cheapest = C.mean(0).min()
    for i in order:
        lines.append(
            f"\\texttt{{{names[i]}}} & {Q[:, i].mean():.3f} & {C[:, i].mean():.2e} & "
            f"{C[:, i].mean() / cheapest:.1f} & {100 * share[i]:.1f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}"]
    write("\n".join(lines), "res_pool")


if __name__ == "__main__":
    table_pool()
    table_rq1()
    table_rq2()
    table_rq3()
