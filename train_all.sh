#!/bin/bash
for s in 0 1 2 3 4; do
  f="ppo_s${s}_x1.6_wait.zip"; [ -f "$f" ] && continue
  python3 train.py --algo ppo --seed $s --reward wait --switch-penalty 0.05 --forced-penalty 0
done
for s in 0 1 2 3 4; do
  f="dqn_s${s}_x1.6_queue.zip"; [ -f "$f" ] && continue
  python3 train.py --algo dqn --seed $s --reward queue
done
