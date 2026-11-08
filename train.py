import os, argparse
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from traffic_env import TrafficEnv

CFG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.sumocfg")

if __name__ == "__main__":      # required because of subprocess environments
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", choices=["dqn", "ppo", "a2c"], default="ppo")
    ap.add_argument("--steps", type=int, default=150_000)    # total env decisions
    ap.add_argument("--n-envs", type=int, default=4)         # PPO / A2C only
    ap.add_argument("--seed", type=int, default=0)           # training seed
    ap.add_argument("--demand-scale", type=float, default=1.6)
    ap.add_argument("--reward", choices=["queue", "wait"], default="queue")
    ap.add_argument("--switch-penalty", type=float, default=0.0)
    ap.add_argument("--forced-penalty", type=float, default=0.1)
    args = ap.parse_args()

    # explicit .zip: SB3 would not add it itself because of the dot in "x1.6"
    name = f"{args.algo}_s{args.seed}_x{args.demand_scale:g}_{args.reward}.zip"
    kw = dict(demand_scale=args.demand_scale, reward_mode=args.reward,
              switch_penalty=args.switch_penalty, forced_penalty=args.forced_penalty)

    try:
        import tensorboard  # noqa: F401
        tb = "./tb"
    except ImportError:
        tb = None   # pip install tensorboard to get learning curves

    policy_kwargs = (dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
                     if args.algo != "dqn" else dict(net_arch=[256, 256]))

    if args.algo == "dqn":
        env = TrafficEnv(CFG, **kw)
        model = DQN("MlpPolicy", env, device="cpu", seed=args.seed,
                    learning_rate=5e-4, buffer_size=100_000, learning_starts=2000,
                    batch_size=128, gamma=0.99, exploration_fraction=0.4,
                    exploration_final_eps=0.02, target_update_interval=1000,
                    policy_kwargs=policy_kwargs, tensorboard_log=tb, verbose=1)
    else:
        env = make_vec_env(lambda: TrafficEnv(CFG, **kw), n_envs=args.n_envs,
                           seed=args.seed, vec_env_cls=SubprocVecEnv)
        if args.algo == "ppo":
            model = PPO("MlpPolicy", env, device="cpu", seed=args.seed,
                        n_steps=512, batch_size=128, n_epochs=10, learning_rate=3e-4,
                        gamma=0.99, ent_coef=0.02, policy_kwargs=policy_kwargs,
                        tensorboard_log=tb, verbose=1)
        else:
            model = A2C("MlpPolicy", env, device="cpu", seed=args.seed,
                        n_steps=64, learning_rate=7e-4, gamma=0.99,
                        ent_coef=0.02, policy_kwargs=policy_kwargs,
                        tensorboard_log=tb, verbose=1)

    model.learn(total_timesteps=args.steps, tb_log_name=name[:-4])
    model.save(name)
    print("Saved", name)
    env.close()
