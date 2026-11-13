# bolt-lite3-dog-hist-drop-215M (13 Nov; a starting point for training, not a robot brain)

Kaggle CPU lite3-dog-38: the Lite3 robot brain hist-196M trained on for 19M steps with 5% of control steps dropped
(`train_imitate.py --drop-p 0.05`, lr 3e-5). The canter entered from a trot on DEEP's model (plan
trot:6,canter:8,trot:4,canter:6,walk:4, 48 tries), falls, hist-196M -> this: 3-6 ms 0 -> 1; 10-19 ms 3 -> 0; turning
left 0.8 at 10-19 ms 10 -> 2; 5% of steps missed 19 -> 8. Better, not solved; its other tests have not been run.
Simulation only. Motion data: CC BY-NC 4.0.
