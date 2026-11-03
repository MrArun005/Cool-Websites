# bolt-lite3-dog-87.3M (simulation): the DEEP Robotics Lite3 robot brain (4 Nov, replaces 63.8M)

The final checkpoint of the same run as 63.8M (Kaggle CPU bolt-lite3-dog-4: 61.8M + corrected leg shapes, 0-60 ms
delay, moving sit/lie starts, steering). The run's own "best" pick was 63.8M, chosen by our exam score, which has no
delay in it; on DEEP's model the later checkpoints hold up better when commands arrive late, which a real robot always
has. Picked on DEEP's model with realistic delay:

| DEEP's MJCF, estimated v/h | 63.8M | **87.3M** |
|---|---|---|
| 41 s routine with 3-6 ms delay, falls (6 tries) | 3/6 (all in the canter) | **1/6** (canter) |
| trot -> sit with 5-15 ms delay (24 tries) | 4/24 | **2/24** |
| trot -> sit with 5-15 ms delay (12 tries, separate run) | 4/12 | **1/12** |
| 41 s routine, no delay | 0/3 | 1/6 |
| fall recovery (side, back, tipped, shove) | 20/20 | 20/20 |
| sit / lie from a moving walk or trot (our model) | 0/48 falls | 0/48 falls |
| our exam: mean tracking | 0.803 | 0.780 |

Also tried (weight_space.py): 0.7/0.3 and 0.5/0.5 blends of 63.8M and 87.3M, and an average of four checkpoints:
none beat 87.3M under delay (5-15 ms 3-6/12 and 6/24). Open: the canter under delay. Simulation only. CC BY-NC clips.
