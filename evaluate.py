import os, re, json, argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import traci
from traffic_env import TrafficEnv

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "config.sumocfg")
COLS = ["halting", "wait", "maxwait", "switch", "forced", "arrived", "left"]
BASE_NAMES = {"fixed": "fixed-time", "greedy": "greedy", "maxpressure": "max-pressure"}


def fixed_policy(env, obs, k):            # 40 s green per phase
    if env.time_in_green >= 40:
        return (env.current + 1) % len(env.green_phases)
    return env.current


def greedy_policy(env, obs, k):           # most halted cars (obs is scaled by lane capacity)
    halt = obs[:len(env.lanes)]
    scores = [halt[idx].sum() for idx in env.phase_lane_idx]
    return int(np.argmax(scores))


def maxpressure_policy(env, obs, k):
    scores = []
    for pl in env.phase_links:
        scores.append(sum(traci.lane.getLastStepVehicleNumber(i) -
                          traci.lane.getLastStepVehicleNumber(o) for i, o in pl))
    return int(np.argmax(scores))


def run(policy, env, seeds, verbose=False):
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        done, k, halts, waits, info = False, 0, [], [], {}
        while not done:
            obs, r, term, trunc, info = env.step(policy(env, obs, k))
            halts.append(info["halting"]); waits.append(info["wait"])
            k += 1
            done = term or trunc
        rows.append((np.mean(halts), np.mean(waits), info["max_wait"], info["switches"],
                     info["forced"], info["arrived"], info["unserved"]))
        if verbose:
            print(f"   seed {seed}: wait={rows[-1][1]:.0f}s arrived={rows[-1][5]} "
                  f"left={rows[-1][6]}", flush=True)
    return np.array(rows)


def load_agent(path):
    import torch
    torch.set_num_threads(1)              # avoid CPU oversubscription with parallel workers
    from stable_baselines3 import DQN, PPO, A2C
    base = os.path.basename(path).lower()
    cls = next((c for k, c in (("dqn", DQN), ("ppo", PPO), ("a2c", A2C)) if base.startswith(k)), None)
    if cls is None:
        raise ValueError(f"Can't tell the algorithm from '{path}' (name must start with dqn/ppo/a2c)")
    model = cls.load(path, device="cpu")
    return lambda e, o, k: int(model.predict(o, deterministic=True)[0])


def eval_job(job):
    spec, scale, seeds, verbose = job
    env = TrafficEnv(CFG, demand_scale=scale)
    builtin = {"fixed": fixed_policy, "greedy": greedy_policy, "maxpressure": maxpressure_policy}
    policy = builtin[spec] if spec in builtin else load_agent(spec)
    rows = run(policy, env, seeds, verbose)
    env.close()
    return spec, rows


def sd(x):
    return np.std(x, axis=0, ddof=1) if len(x) > 1 else np.zeros(x.shape[1:])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=[], help="model files, e.g. ppo_s*_x1.6_queue.zip")
    ap.add_argument("--demand-scale", type=float, default=1.6)
    ap.add_argument("--seeds", nargs=2, type=int, default=[501, 511], metavar=("START", "END"),
                    help="demand seeds range(START, END). Default 501-510 = VALIDATION. "
                         "Use 2001 2021 only for the final reported run.")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--rerun-baselines", action="store_true")
    args = ap.parse_args()
    scale = args.demand_scale
    seeds = list(range(args.seeds[0], args.seeds[1]))
    stag = f"{seeds[0]}-{seeds[-1]}"

    # ---- baselines: cached per scale and per seed range ----
    cache_file = os.path.join(HERE, f"baselines_cache_x{scale:g}_s{stag}.json")
    base = {}
    if os.path.exists(cache_file) and not args.rerun_baselines:
        with open(cache_file) as f:
            saved = json.load(f)
        if saved.get("seeds") == seeds:
            base = {k: np.array(v) for k, v in saved["results"].items()}
            print("Loaded cached baselines")

    models = [m[:-4] if m.endswith(".zip") else m for m in args.model]
    jobs = [(s, scale, seeds, args.workers == 1) for s in BASE_NAMES if BASE_NAMES[s] not in base]
    jobs += [(m, scale, seeds, args.workers == 1) for m in models]

    results = {}
    if args.workers > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=args.workers,
                                 mp_context=mp.get_context("spawn")) as ex:
            for spec, rows in ex.map(eval_job, jobs):
                results[spec] = rows
                print(f"done: {os.path.basename(spec)}", flush=True)
    else:
        for job in jobs:
            print(f"--- {os.path.basename(job[0])}", flush=True)
            spec, rows = eval_job(job)
            results[spec] = rows

    for s, name in BASE_NAMES.items():
        if s in results:
            base[name] = results[s]
    with open(cache_file, "w") as f:
        json.dump({"seeds": seeds, "results": {k: v.tolist() for k, v in base.items()}}, f)

    # ---- group agents by algorithm (training seed removed from the name) ----
    groups = {}
    for m in models:
        g = re.sub(r"_s\d+", "", os.path.basename(m))
        groups.setdefault(g, []).append(results[m])

    # ---- table 1: absolute metrics ----
    print(f"\nDemand scale x{scale:g}, demand seeds {stag} ({len(seeds)})")
    print("baselines: mean over demand seeds | agents: mean ± std across training seeds\n")
    print(f"{'policy':<26}{'n':>3}" + "".join(f"{c:>13}" for c in COLS))
    for name, r in base.items():
        print(f"{name:<26}{1:>3}" + "".join(f"{v:>13.1f}" for v in r.mean(axis=0)))
    for g, rl in groups.items():
        M = np.array([r.mean(axis=0) for r in rl])           # models x cols
        m, s = M.mean(axis=0), sd(M)
        print(f"{g:<26}{len(rl):>3}" + "".join(f"{f'{a:.1f}±{b:.1f}':>13}" for a, b in zip(m, s)))

    # ---- table 2: paired differences (same demand seed), agent minus baseline ----
    ci = {c: COLS.index(c) for c in COLS}
    for bname, brows in base.items():
        print(f"\nPaired difference vs {bname}  (negative = better, except 'arrived')")
        print(f"{'policy':<26}{'d_wait':>20}{'d_left':>14}{'d_maxwait':>14}{'d_switch':>14}"
              f"{'win%wait':>10}{'win%left':>10}")
        for g, rl in groups.items():
            D = np.array([r - brows for r in rl])             # models x seeds x cols
            pm = D.mean(axis=1)                                # per-model mean difference
            m, s = pm.mean(axis=0), sd(pm)
            pct = 100 * m[ci["wait"]] / brows[:, ci["wait"]].mean()
            cell = lambda c: f"{m[ci[c]]:+.1f}±{s[ci[c]]:.1f}"
            wins = lambda c: 100 * (D[:, :, ci[c]] < 0).mean()
            print(f"{g:<26}{cell('wait') + f' ({pct:+.0f}%)':>20}{cell('left'):>14}"
                  f"{cell('maxwait'):>14}{cell('switch'):>14}"
                  f"{wins('wait'):>10.0f}{wins('left'):>10.0f}")

    out = os.path.join(HERE, f"results_x{scale:g}_s{stag}.json")
    with open(out, "w") as f:
        json.dump({"seeds": seeds, "scale": scale, "columns": COLS,
                   "baselines": {k: v.tolist() for k, v in base.items()},
                   "agents": {os.path.basename(m): results[m].tolist() for m in models}}, f)
    print(f"\nRaw per-seed results saved to {out}")
