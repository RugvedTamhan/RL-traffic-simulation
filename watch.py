import os, argparse
import traci
from traffic_env import TrafficEnv
from evaluate import fixed_policy, greedy_policy, maxpressure_policy, load_agent

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.sumocfg")

ap = argparse.ArgumentParser()
ap.add_argument("--policy", choices=["fixed", "greedy", "maxpressure", "agent"], default="agent")
ap.add_argument("--model", default="ppo_s0_x1.6_wait.zip")
ap.add_argument("--seed", type=int, default=501)      # same seed = same traffic for every policy
ap.add_argument("--demand-scale", type=float, default=1.6)
args = ap.parse_args()

policy = {"fixed": fixed_policy, "greedy": greedy_policy,
          "maxpressure": maxpressure_policy}.get(args.policy)
if policy is None:
    policy = load_agent(args.model)

env = TrafficEnv(CFG, gui=True, demand_scale=args.demand_scale)
obs, info = env.reset(seed=args.seed)
print("Demand this episode:", info["probs"])
print(f"{'t(s)':>6} {'action':>7} {'switched':>9} {'halting':>8}")

done, k = False, 0
while not done:
    before = env.current
    action = policy(env, obs, k)
    obs, r, term, trunc, info = env.step(action)
    switched = env.current != before
    t = traci.simulation.getTime()
    print(f"{t:>6.0f} {action:>7} {'YES' if switched else '':>9} {info['halting']:>8.0f}")
    k += 1
    done = term or trunc

print(f"\n{args.policy}: {info['switches']} switches, {info['forced']} forced, "
      f"max wait {info['max_wait']:.0f}s, arrived {info['arrived']}, left {info['unserved']}")
try:
    env.close()
except Exception:
    pass
