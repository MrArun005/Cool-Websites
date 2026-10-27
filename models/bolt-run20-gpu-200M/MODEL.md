# Bolt fast-running brain, run 20 on a Kaggle GPU (200M steps, MJX + Brax PPO) — speed reference only

Retired gait: it runs fast but not like a dog (style score 0.11), so the product uses the dog-imitation brains instead.

Exam (`run_env.exam`, randomized physics, `mjx_run.MjxPolicy("params.pkl")`):
- Speed kept: 1 m/s 98%, 2 m/s 98%, 3 m/s 96% (about 2.9 m/s) with no falls; 4 m/s falls.
- Turning at 3 and 5 rad/s: falls. Survives a 1.5 m/s sideways shove.
- MIT Mini Cheetah, a similar-sized robot, reaches 3.9 m/s; Bolt reaches about 2.9 m/s (74%).
