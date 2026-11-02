# bolt-lite3-selfright-28M (simulation): rolls a fallen DEEP Robotics Lite3 back onto its belly

Kaggle CPU run bolt-lite3-selfright-pilot (2 Nov, 635 min, 28.3M steps, from scratch): train_getup.py --body lite3
--goal selfright --hard (kicks, drops, ice, a second kick), 0-40 ms delay; getup_env sizes for the Lite3 (heights x0.80,
kicks x0.90). The motion brain's getup clip then stands it up, as on the Go2 (deploy/recovery.py).

Exam at 28M (10 tries per start): righted from its side 10/10 in 0.64 s, from its back 10/10 in 0.65 s, tipped 10/10 in
0.63 s, standing 10/10. It first reached 10/10 everywhere at 5M (0.76-0.83 s); this later checkpoint is as reliable and
faster. Win line (>= 0.5 from side and back): met. Simulation only.
