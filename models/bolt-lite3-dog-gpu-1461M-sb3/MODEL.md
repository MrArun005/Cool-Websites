# bolt-lite3-dog-gpu-1461M-sb3 (18 Nov; a starting point for CPU training, not a robot brain)

The Lite3 robot brain bolt-lite3-dog-gpu-1461M written as an SB3 folder so the CPU trainer can go on from it
(`train_imitate.py --init-from <this> --history 3 --blind-vel`). The brain is identical (round trip through
`mjx_imitate.from_sb3`: differences 0); the value network lost its 8 privileged inputs. Simulation only.
