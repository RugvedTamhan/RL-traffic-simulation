import os, sys, random, tempfile
import numpy as np
import gymnasium as gym
from gymnasium import spaces

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
else:
    sys.exit("Please set SUMO_HOME")
import traci

# (flow id, from edge, to edge, min prob, max prob) -- edge IDs must match YOUR network
FLOWS = [
    ("f_E0_E1", "E0",  "E1", 0.15, 0.25),
    ("f_E0_E3", "E0",  "E3", 0.03, 0.08),
    ("f_E4_E3", "-E4", "E3", 0.03, 0.10),
    ("f_E3_E4", "-E3", "E4", 0.03, 0.10),
    ("f_E3_E1", "-E3", "E1", 0.03, 0.07),
    ("f_E4_E1", "-E4", "E1", 0.03, 0.07),
]
CAR_SPACE = 7.5   # metres one queued car occupies (5 m car + 2.5 m gap)


def write_random_routes(path, rng, scale=1.0):
    probs = {}
    lines = ["<routes>",
             '  <vType id="car" accel="2.6" decel="4.5" length="5" maxSpeed="13.9" sigma="0.5"/>']
    for fid, a, b, lo, hi in FLOWS:
        p = min(rng.uniform(lo, hi) * scale, 0.9)      # a flow can't exceed ~1 car/s
        probs[fid] = round(p, 3)
        lines.append(f'  <flow id="{fid}" type="car" begin="0" end="3600" probability="{p:.3f}" '
                     f'from="{a}" to="{b}" departLane="best" departSpeed="max"/>')
    lines.append("</routes>")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    return probs


class TrafficEnv(gym.Env):
    def __init__(self, cfg, gui=False, delta_time=10, yellow_time=3, all_red_time=2,
                 min_green=10, max_green=45, episode_len=3600,
                 switch_penalty=0.0, forced_penalty=0.1, reward_mode="queue",
                 reward_scale=100.0, demand_scale=1.0):
        super().__init__()
        assert reward_mode in ("queue", "wait")
        self.cfg, self.gui = cfg, gui
        self.delta_time, self.yellow_time, self.all_red_time = delta_time, yellow_time, all_red_time
        self.min_green, self.max_green = min_green, max_green
        self.episode_len = episode_len
        self.demand_scale = demand_scale
        self.switch_penalty, self.forced_penalty = switch_penalty, forced_penalty
        self.reward_mode = reward_mode
        self.reward_scale = reward_scale * demand_scale   # only used by the "wait" reward
        self.running = False
        self.step_cost = 0.0
        self.route_file = os.path.join(tempfile.gettempdir(), f"routes_{os.getpid()}.rou.xml")

        # Start once to read the signal structure, then close
        self._start(seed=0)
        self.tls = traci.trafficlight.getIDList()[0]
        logic = traci.trafficlight.getAllProgramLogics(self.tls)[0]
        self.program_id = logic.programID
        self.n_signals = len(traci.trafficlight.getRedYellowGreenState(self.tls))
        self.green_phases = [i for i, p in enumerate(logic.phases)
                             if "y" not in p.state and any(c in "Gg" for c in p.state)]
        for g in self.green_phases:
            assert "y" in logic.phases[g + 1].state, f"Phase {g + 1} is not a yellow phase"

        self.lanes = list(dict.fromkeys(traci.trafficlight.getControlledLanes(self.tls)))
        self.capacity = np.array([max(int(traci.lane.getLength(l) // CAR_SPACE), 1)
                                  for l in self.lanes], dtype=np.float32)
        self.total_cap = float(self.capacity.sum())

        links = traci.trafficlight.getControlledLinks(self.tls)
        self.phase_lane_idx = []          # per action: indices into self.lanes it serves
        self.phase_links = []             # per action: (in_lane, out_lane) pairs it serves
        for g in self.green_phases:
            state = logic.phases[g].state
            idxs = {self.lanes.index(links[i][0][0])
                    for i, ch in enumerate(state) if ch in "Gg" and links[i]}
            self.phase_lane_idx.append(sorted(idxs))
            self.phase_links.append([(links[i][0][0], links[i][0][1])
                                     for i, ch in enumerate(state) if ch in "Gg" and links[i]])
        self._close()

        n = len(self.green_phases)
        self.action_space = spaces.Discrete(n)
        # halting/lane + count/lane + current phase one-hot + time in phase + longest wait/phase
        self.observation_space = spaces.Box(0.0, 1.0, shape=(2 * len(self.lanes) + 2 * n + 1,),
                                            dtype=np.float32)

    # ---------- SUMO helpers ----------
    def _start(self, seed):
        self.episode_probs = write_random_routes(self.route_file, random.Random(seed),
                                                 self.demand_scale)
        cmd = ["sumo-gui" if self.gui else "sumo", "-c", self.cfg,
               "--route-files", self.route_file, "--seed", str(seed),
               "--no-warnings", "--no-step-log", "--time-to-teleport", "-1"]
        if self.gui:
            cmd += ["--start", "--quit-on-end", "--delay", "50"]
        traci.start(cmd)
        self.running = True

    def _close(self):
        if self.running:
            traci.close()
            self.running = False

    def _veh_cost(self):
        """Vehicles on the controlled lanes right now, as a fraction of total lane capacity."""
        return sum(traci.lane.getLastStepVehicleNumber(l) for l in self.lanes) / self.total_cap

    def _sim(self, n):
        for _ in range(n):
            traci.simulationStep()
            self.time_in_green += 1
            self.arrived += traci.simulation.getArrivedNumber()
            self.step_cost += self._veh_cost()      # integrates over every simulated second

    def _lane_vals(self, fn):
        return np.array([fn(l) for l in self.lanes], dtype=np.float32)

    def _total_wait(self):
        return float(self._lane_vals(traci.lane.getWaitingTime).sum())

    def _phase_halt(self, idx):
        return sum(traci.lane.getLastStepHaltingNumber(self.lanes[i])
                   for i in self.phase_lane_idx[idx])

    def _longest_wait(self):
        """Longest time any single vehicle has currently been standing still."""
        return max((traci.vehicle.getWaitingTime(v) for v in traci.vehicle.getIDList()),
                   default=0.0)

    def _phase_max_wait(self):
        """Per action: longest current wait (scaled by 60 s) among vehicles on the lanes it serves."""
        out = []
        for idxs in self.phase_lane_idx:
            w = 0.0
            for i in idxs:
                for v in traci.lane.getLastStepVehicleIDs(self.lanes[i]):
                    w = max(w, traci.vehicle.getWaitingTime(v))
            out.append(min(w / 60.0, 1.0))
        return np.array(out, dtype=np.float32)

    def _obs(self):
        halt = np.clip(self._lane_vals(traci.lane.getLastStepHaltingNumber) / self.capacity, 0, 1)
        cnt = np.clip(self._lane_vals(traci.lane.getLastStepVehicleNumber) / self.capacity, 0, 1)
        onehot = np.zeros(len(self.green_phases), dtype=np.float32)
        onehot[self.current] = 1.0
        t = np.array([min(self.time_in_green / 60.0, 1.0)], dtype=np.float32)
        return np.concatenate([halt, cnt, onehot, t, self._phase_max_wait()]).astype(np.float32)

    def _set_green(self, idx):
        traci.trafficlight.setProgram(self.tls, self.program_id)   # undo any all-red override
        traci.trafficlight.setPhase(self.tls, self.green_phases[idx])
        traci.trafficlight.setPhaseDuration(self.tls, 9999)        # stop SUMO auto-advancing

    # ---------- Gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            # drawn from the env's own RNG, so the training seed controls the whole demand
            # sequence; offset keeps training seeds away from evaluation seeds
            seed = 100_000 + int(self.np_random.integers(900_000))
        self._close()
        self._start(seed)
        self.current, self.time_in_green, self.arrived = 0, 0, 0
        self.n_switches, self.n_forced, self.max_wait = 0, 0, 0.0
        self._set_green(self.current)
        self._sim(self.delta_time)
        self.prev_wait = self._total_wait()
        self.step_cost = 0.0
        return self._obs(), {"probs": self.episode_probs}

    def step(self, action):
        action = int(action)
        forced_now = 0

        # Safety rule: max green. If vehicles wait elsewhere, force a switch.
        if action == self.current and self.time_in_green >= self.max_green:
            others = [(self._phase_halt(i), i) for i in range(len(self.green_phases))
                      if i != self.current]
            best_halt, best = max(others)
            if best_halt > 0:
                action = best
                forced_now = 1
                self.n_forced += 1

        # Safety rule: min green. Switch requests that come too early are ignored.
        switched = action != self.current and self.time_in_green >= self.min_green
        if switched:
            # 1) yellow (the network's own yellow phase for the current green)
            traci.trafficlight.setPhase(self.tls, self.green_phases[self.current] + 1)
            traci.trafficlight.setPhaseDuration(self.tls, 9999)
            self._sim(self.yellow_time)
            # 2) all-red clearance
            if self.all_red_time > 0:
                traci.trafficlight.setRedYellowGreenState(self.tls, "r" * self.n_signals)
                self._sim(self.all_red_time)
            # 3) new green, guaranteed for min_green seconds
            self.current = action
            self._set_green(action)
            self.time_in_green = 0
            self._sim(self.min_green)
            self.n_switches += 1
        else:
            self._sim(self.delta_time)

        wait = self._total_wait()
        if self.reward_mode == "queue":
            base = -self.step_cost / self.delta_time       # ~ average fullness over this step
        else:
            base = (self.prev_wait - wait) / self.reward_scale
        reward = base - self.switch_penalty * switched - self.forced_penalty * forced_now
        self.prev_wait = wait
        self.step_cost = 0.0
        self.max_wait = max(self.max_wait, self._longest_wait())

        halting = float(self._lane_vals(traci.lane.getLastStepHaltingNumber).sum())
        truncated = (traci.simulation.getTime() >= self.episode_len
                     or traci.simulation.getMinExpectedNumber() == 0)
        info = {"halting": halting, "wait": wait, "arrived": self.arrived,
                "max_wait": self.max_wait, "switches": self.n_switches,
                "forced": self.n_forced,
                "unserved": traci.simulation.getMinExpectedNumber()}
        return self._obs(), reward, False, truncated, info

    def close(self):
        self._close()
