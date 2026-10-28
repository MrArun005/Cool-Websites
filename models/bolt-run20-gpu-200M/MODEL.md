# Bolt fast-running brain, run 20 on a Kaggle GPU (200M steps, MJX + Brax PPO) — speed reference only

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: mjx_run.py's MOTOR is torque=33.5 (file added 27 Oct 11:43, after fd5199b).

Retired gait: it runs fast but not like a dog (style score 0.11), so the product uses the dog-imitation brains instead.

Exam (`run_env.exam`, randomized physics, `mjx_run.MjxPolicy("params.pkl")`):
- Speed kept: 1 m/s 98%, 2 m/s 98%, 3 m/s 96% (about 2.9 m/s) with no falls; 4 m/s falls.
- Turning at 3 and 5 rad/s: falls. Survives a 1.5 m/s sideways shove.
- MIT Mini Cheetah, a similar-sized robot, reaches 3.9 m/s; Bolt reaches about 2.9 m/s (74%).
