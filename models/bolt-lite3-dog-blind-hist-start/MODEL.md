# bolt-lite3-dog-blind-hist-start (10 Nov; a starting point for training, not a robot brain)

The blind Lite3 brain with 3 frames of input history (see bolt-lite3-dog-blind-hist-candidate), copied for 25 more
rounds with a value head, and written as an SB3 folder so PPO can go on from it:
`train_imitate.py --init-from <this> --history 3 --blind-vel`. Fresh-seed steered canter (24 tries): 0.349 with 6
falls (the candidate: 0.385 with 5; within the noise). Simulation only. Motion data: CC BY-NC 4.0.
