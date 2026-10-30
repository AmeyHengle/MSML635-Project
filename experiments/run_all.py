"""Run RQ1, RQ2, and RQ3 and write results to ``results/<source>/``.

    uv run python experiments/run_all.py                 # everything, full data
    uv run python experiments/run_all.py rq1 rq3         # a subset
    uv run python experiments/run_all.py --force rq2     # recompute, ignore cache
    RB_SOURCE=fixture uv run python experiments/run_all.py   # quick 5k test

Every (experiment, setting, seed) cell is saved to ``results/<source>/cells/`` as
soon as it finishes; rerunning skips cells that already exist. The merged
``rq1.csv`` / ``rq2.csv`` / ``rq3.csv`` are rebuilt from the cells each time.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from common import (
    LAMBDA_GRID,
    LAMBDA_REF,
    RESULTS,
    SEEDS,
    SOURCE,
    build_policies,
    build_setup,
    cached_cell,
    check_or_write_meta,
    fit_supervised,
    make_env,
    make_heldout_env,
)

from router_bandit.baselines import OraclePolicy
from router_bandit.envs import ShiftSchedule
from router_bandit.evaluation import (
    doubly_robust,
    fit_reward_model,
    heldout_summary,
    ips,
    performance_gap_recovered,
    pool_reference_arms,
    recovery_time,
    routing_errors,
    run_policy,
    snips,
)
from router_bandit.evaluation.shift_calibration import (
    ChosenStrength,
    calibrate_degradation,
    calibrate_price,
    flip_rate,
    most_selected_arm,
    pull_rate,
    scenario_schedules,
)
from router_bandit.policies import (
    DiscountedLinUCB,
    EpsilonGreedy,
    LinUCB,
    UniformRandom,
)


# --------------------------------------------------------------------------- #
# RQ1 -- stationary: how close do bandits get to full information?
# --------------------------------------------------------------------------- #
def rq1(setup, force: bool = False) -> pd.DataFrame:
    def cell(lam: float, seed: int) -> pd.DataFrame:
        rows = []
        env, stream = make_env(setup, lam, seed)
        held_env, held_stream = make_heldout_env(setup, lam, seed)
        # Reference arms are properties of the pool, not of this lambda. Their
        # quality is measured on the same held-out queries as the router.
        weak, strong = pool_reference_arms(setup.data.cost.mean(0), setup.data.quality.mean(0))
        q_held = held_stream.quality_matrix()
        q_weak = float(q_held[:, weak].mean())
        q_strong = float(q_held[:, strong].mean())

        for policy in build_policies(setup, stream, env, lam, seed):
            res = run_policy(env, policy)
            # The oracle reads the env it was built with. Rebind it so the
            # frozen replay scores the held-out queries, not the stream's.
            frozen = OraclePolicy(held_env.truth) if policy.name == "Oracle" else policy
            held = run_policy(held_env, frozen, reset=False, update=False)
            errs = routing_errors(res.actions, res.oracle_actions, setup.data.cost.mean(0))
            dollar = setup.data.cost[held_stream.order, held.actions]
            rows.append({
                "lam": lam, "seed": seed, **res.summary(),
                "mean_dollar_cost": float(setup.data.cost[stream.order, res.actions].mean()),
                **errs,
                **heldout_summary(held.rewards, held.qualities, dollar, seed=seed),
                "pgr": performance_gap_recovered(held.mean_quality, q_weak, q_strong),
                # Frozen select, not the learning pass: this is the latency of
                # the policy that would actually be served.
                "select_p50_s": held.select_p50,
                "select_p99_s": held.select_p99,
            })
        return pd.DataFrame(rows)

    frames = [
        cached_cell("rq1", f"lam={lam:g}_seed={seed}",
                    lambda lam=lam, seed=seed: cell(lam, seed), force=force)
        for lam in LAMBDA_GRID
        for seed in SEEDS
    ]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# RQ2 -- shifts: do bandits adapt faster than a static supervised router?
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rq2Scenario:
    """One shift, plus the arm and strength its rows should report."""

    schedule: ShiftSchedule
    shocked_arm: int | None = None
    strength: float | None = None
    router_flip: float | None = None


def calibrate_router_shocks(setup, stream, t0: int) -> tuple[int, ChosenStrength, ChosenStrength]:
    """Arm the supervised router uses, and the S2/S4 strengths that move it.

    Measured on the seed-0 stationary stream at the reference lambda. The same
    arm and strengths are then used for every RQ2 seed.
    """
    router = fit_supervised(setup, stream, LAMBDA_REF)
    preds = router.predict_rewards(stream.contexts())
    arm = most_selected_arm(preds.argmax(axis=1), setup.n_arms)
    price = calibrate_price(preds, stream.cost_matrix(), arm, t0, LAMBDA_REF)
    degrade = calibrate_degradation(preds, stream.quality_matrix(), arm, t0)
    return arm, price, degrade


def _strength_note(chosen: ChosenStrength) -> str:
    relation = "reaches" if chosen.reached_target else "stays under"
    return f"{chosen.strength:g} ({relation} 25%, router flip {chosen.flip_rate:.3f})"


def shift_schedules(
    setup,
    t0: int,
    *,
    shocked_arm: int,
    price: ChosenStrength,
    degrade: ChosenStrength,
) -> dict[str, Rq2Scenario]:
    """The four scenarios.

    S1 still withholds the oracle's most-used arm until ``t0``, and S3 still
    reweights the query mix. S2 and S4 shock the arm the static supervised
    router actually selects. Shocking the oracle's arm instead misses routers
    that over-escalate, and those two scenarios then barely move reward.
    """
    oracle = setup.data.oracle(LAMBDA_REF)
    usage = np.bincount(oracle, minlength=setup.n_arms)
    workhorse = int(np.argmax(usage))
    raw = scenario_schedules(
        t0,
        oracle_arm=workhorse,
        shocked_arm=shocked_arm,
        multiplier=price.strength,
        severity=degrade.strength,
    )
    return {
        "S0-stationary": Rq2Scenario(raw["S0-stationary"]),
        "S1-new-model": Rq2Scenario(raw["S1-new-model"], shocked_arm=workhorse),
        "S2-price-change": Rq2Scenario(
            raw["S2-price-change"],
            shocked_arm=shocked_arm,
            strength=price.strength,
            router_flip=price.flip_rate,
        ),
        "S3-query-mix": Rq2Scenario(raw["S3-query-mix"]),
        "S4-silent-degrade": Rq2Scenario(
            raw["S4-silent-degrade"],
            shocked_arm=shocked_arm,
            strength=degrade.strength,
            router_flip=degrade.flip_rate,
        ),
    }


def _shock_columns(actions: np.ndarray, t0: int, oracle_flip: float, spec: Rq2Scenario) -> dict:
    """Oracle flip is a property of the scenario; pull rates are this policy's."""
    arm = spec.shocked_arm
    if arm is None:
        pre = post = np.nan
    else:
        pre = pull_rate(actions, arm, 0, t0)
        post = pull_rate(actions, arm, t0, actions.size)
    return {
        "shocked_arm": np.nan if arm is None else arm,
        "shift_strength": np.nan if spec.strength is None else spec.strength,
        "oracle_flip_rate": oracle_flip,
        "router_flip_rate": np.nan if spec.router_flip is None else spec.router_flip,
        "pull_pre": pre,
        "pull_post": post,
    }


def rq2(setup, force: bool = False) -> pd.DataFrame:
    probe_env, probe_stream = make_env(setup, LAMBDA_REF, 0)
    t0 = probe_env.n_rounds // 2
    shocked_arm, price, degrade = calibrate_router_shocks(setup, probe_stream, t0)
    print(
        f"  calibration at lambda={LAMBDA_REF:g}, t0={t0}: shocked arm {shocked_arm}, "
        f"S2 x{_strength_note(price)}, S4 severity {_strength_note(degrade)}",
        flush=True,
    )
    schedules = shift_schedules(
        setup, t0, shocked_arm=shocked_arm, price=price, degrade=degrade
    )

    def cell(name: str, seed: int) -> pd.DataFrame:
        rows = []
        spec = schedules[name]
        env, stream = make_env(setup, LAMBDA_REF, seed, spec.schedule)
        # Same query, stationary prices and quality, versus this scenario's oracle.
        base_oracle = setup.data.oracle(LAMBDA_REF)[stream.order]
        oracle_flip = flip_rate(base_oracle, env.truth.oracle_actions(), t0)
        k, d = setup.n_arms, stream.dim
        policies = [
            LinUCB(k, d, alpha=0.3, seed=seed),
            DiscountedLinUCB(k, d, alpha=0.3, gamma=0.9995, seed=seed),
            EpsilonGreedy(k, d, epsilon=0.1, seed=seed),
            fit_supervised(setup, stream, LAMBDA_REF),
        ]
        for policy in policies:
            res = run_policy(env, policy)
            # Post-shift regret stays the headline. Departure and recovery are
            # missing, not zero, when the policy never leaves the pre-shift band
            # or never returns to it.
            rec = recovery_time(res.rewards, t0)
            rows.append({
                "scenario": name, "seed": seed, "t0": t0, **res.summary(),
                "pre_reward": float(res.rewards[:t0].mean()),
                "post_reward": float(res.rewards[t0:].mean()),
                "post_regret": float(res.instant_regret[t0:].sum()),
                "departed_at": np.nan if rec.departed_at is None else rec.departed_at,
                "rounds_to_recover": (
                    np.nan if rec.rounds_to_recover is None else rec.rounds_to_recover
                ),
                **_shock_columns(res.actions, t0, oracle_flip, spec),
            })
        return pd.DataFrame(rows)

    frames = [
        cached_cell("rq2", f"{name}_seed={seed}",
                    lambda name=name, seed=seed: cell(name, seed), force=force)
        for name in schedules
        for seed in SEEDS
    ]
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------- #
# RQ3 -- OPE: can IPS/SNIPS/DR predict a policy's value from logs?
# --------------------------------------------------------------------------- #
def rq3(setup, force: bool = False) -> pd.DataFrame:
    frames = [
        cached_cell("rq3", f"seed={seed}", lambda seed=seed: _rq3_cell(setup, seed), force=force)
        for seed in SEEDS
    ]
    return pd.concat(frames, ignore_index=True)


def _rq3_cell(setup, seed: int) -> pd.DataFrame:
    rows = []
    env, stream = make_env(setup, LAMBDA_REF, seed)
    k, d = setup.n_arms, stream.dim

    loggers = {
        "uniform": UniformRandom(k, seed=seed),
        "eps-greedy(0.3)": EpsilonGreedy(k, d, epsilon=0.3, seed=seed),
        "eps-greedy(0.1)": EpsilonGreedy(k, d, epsilon=0.1, seed=seed),
    }
    targets = {
        "LinUCB": LinUCB(k, d, alpha=0.3, seed=seed + 100),
        "Supervised": fit_supervised(setup, stream, LAMBDA_REF),
        "always-cheapest": None,
    }

    for log_name, log_policy in loggers.items():
        log_run = run_policy(env, log_policy, log=True)
        log = log_run.log
        log.check_loggable()
        q_hat = fit_reward_model(log.contexts, log.actions, log.rewards, k)

        for tgt_name, tgt_policy in targets.items():
            if tgt_policy is None:
                cheapest = int(np.argmin(setup.data.cost.mean(0)))
                target_actions = np.full(len(log), cheapest, dtype=np.int64)
                truth = float(env.truth.reward_matrix()[:, cheapest].mean())
            else:
                tgt_run = run_policy(env, tgt_policy)
                target_actions = tgt_run.actions
                truth = tgt_run.mean_reward  # on-policy ground truth

            ests = [
                ips(log.rewards, log.actions, log.propensities, target_actions),
                snips(log.rewards, log.actions, log.propensities, target_actions),
                doubly_robust(log.rewards, log.actions, log.propensities,
                              target_actions, q_hat),
            ]
            for est in ests:
                rows.append({
                    "seed": seed, "logger": log_name, "target": tgt_name,
                    "estimator": est.estimator, "estimate": est.value,
                    "std_error": est.std_error, "truth": truth,
                    "bias": est.value - truth,
                    "abs_error": abs(est.value - truth),
                    "min_propensity": log.min_propensity,
                })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
def main() -> None:
    args = [a.lower() for a in sys.argv[1:]]
    force = "--force" in args
    which = [a for a in args if not a.startswith("--")] or ["rq1", "rq2", "rq3"]
    unknown = set(which) - {"rq1", "rq2", "rq3"}
    if unknown:
        raise SystemExit(f"Unknown experiment(s): {sorted(unknown)}")

    print(f"source: {SOURCE}  ->  {RESULTS}", flush=True)
    print("building setup ...", flush=True)
    setup = build_setup(seed=0)
    print(f"  data: {setup.scale_note}, {setup.data.n_queries:,} queries, "
          f"{setup.n_arms} arms, stream={len(setup.splits.stream):,}", flush=True)
    check_or_write_meta(setup, force=force)

    for name, fn in (("rq1", rq1), ("rq2", rq2), ("rq3", rq3)):
        if name not in which:
            continue
        start = time.time()
        df = fn(setup, force=force)
        path = RESULTS / f"{name}.csv"
        df.to_csv(path, index=False)
        print(f"{name}: {len(df)} rows -> {path}  [{time.time() - start:.1f}s]", flush=True)


if __name__ == "__main__":
    main()
