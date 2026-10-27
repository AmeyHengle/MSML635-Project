"""Emit LaTeX tables for the report from ``results/*.csv``."""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import LAMBDA_REF, REPO, RESULTS

TABDIR = REPO / "report" / "tables"
TABDIR.mkdir(parents=True, exist_ok=True)

SHORT = {
    "Oracle": "Per-query oracle",
    "Supervised(ridge)": "Supervised (full info)",
    "LinUCB(a=0.3)": "LinUCB",
    "LinTS(v=0.2)": "LinTS",
    "D-LinUCB(a=0.3,g=0.9995)": "Discounted LinUCB",
    "eps-greedy(e=0.1)": "$\\epsilon$-greedy",
    "always-cheapest": "Always cheapest",
    "always-premium": "Always premium",
    "Random": "Random",
}


def pretty(name: str) -> str:
    if name.startswith("ZeroRouter"):
        return "Zero Router"
    return SHORT.get(name, name)


def write(text: str, name: str) -> None:
    (TABDIR / f"{name}.tex").write_text(text)
    print(f"saved -> report/tables/{name}.tex")


def table_rq1() -> None:
    df = pd.read_csv(RESULTS / "rq1.csv")
    ref = df[np.isclose(df["lam"], LAMBDA_REF)]
    agg = ref.groupby("policy").agg(
        reward=("mean_reward", "mean"), reward_sd=("mean_reward", "std"),
        quality=("mean_quality", "mean"), cost=("mean_dollar_cost", "mean"),
        regret=("total_regret", "mean"), match=("oracle_match", "mean"),
        esc=("escalation_rate", "mean"),
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
