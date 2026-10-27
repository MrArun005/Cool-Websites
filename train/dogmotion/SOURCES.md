# Dog motion sources

## The motion capture

**Mode-Adaptive Neural Networks for Quadruped Motion Control** (He Zhang, Sebastian Starke, Taku Komura,
Jun Saito; SIGGRAPH 2018). A real dog (one medium-size dog) in an optical motion-capture suit: 51 BVH files,
60 frames/s, a 27-joint skeleton (spine, neck, head, four 3-segment legs with paw end sites, tail), about
30 minutes of walking, trotting, cantering, turning, sitting, lying, standing up, jumping.

- Project: https://github.com/sebastianstarke/AI4Animation (README, section "SIGGRAPH 2018")
- Download used: https://starke-consult.de/AI4Animation/SIGGRAPH_2018/MotionCapture.zip
  (33,422,409 bytes, last modified 2018-06-02; fetched 2026-10-27)
- **License: Creative Commons Attribution-NonCommercial 4.0 (CC BY-NC 4.0)**,
  https://creativecommons.org/licenses/by-nc/4.0/legalcode. The README adds: "This project is only for
  research or education purposes, and not freely available for commercial use or redistribution."

The raw BVH files are **not** in this repository; `retarget.py --bvh-dir` reads them from wherever the zip
was unpacked. The retargeted clips in `clips/` are derived from this data, so they carry the same terms:
credit the authors above, non-commercial use only. Any brain trained on them, and any page that plays them,
should state that too.

## The clip choices and the retargeting method

**Learning Agile Robotic Locomotion Skills by Imitating Animals** (Xue Bin Peng, Erwin Coumans, Tingnan
Zhang, Tsang-Wei Edward Lee, Jie Tan, Sergey Levine; RSS 2020).
Code: https://github.com/google-research/motion_imitation (Apache-2.0). Its `retarget_motion/data/` holds
the same dog mocap as keypoint text files (with the same CC BY-NC 4.0 notice); our BVH forward kinematics
reproduces those keypoints to 3e-6 m, which checks `bvh.py`. We reuse from it:

- the looping segments it picked (60 Hz frame numbers): pace = D1_007_KAN01_001 (its dog_walk00) 162-201,
  trot = D1_009_KAN01_001 (dog_walk03) 448-481, canter = D1_010_KAN01_001 (dog_run00) 430-459, and the turn
  D1_053_KAN01_001 (dog_walk09) 1085-1124, which it calls "left turn0" but which turns clockwise (right) in
  the data's own frame;
- the keypoint retargeting idea: body frame from the shoulders and hips, paws placed relative to the
  robot's hips, paw height kept absolute so planted paws stay on the floor, then leg IK.

The code here is our own (no code copied).

## Clips in `clips/`

| clip | source file (clip name in the MANN list) | seconds (60 Hz frames) | notes |
|---|---|---|---|
| walk | D1_ex01_KAN01_001 (dog_walk14) | one stride found from 4.0 s (auto, ~0.66 s) | steady ~1.1 m/s walk |
| trot | D1_009_KAN01_001 (dog_walk03) | 448-481 | 1.9 m/s |
| pace | D1_007_KAN01_001 (dog_walk00) | 162-201 | 1.15 m/s |
| canter | D1_010_KAN01_001 (dog_run00) | 430-459 | 3.4 m/s; not in the default training set |
| turn_right | D1_053_KAN01_001 (dog_walk09) | 1085-1124 | 0.9 m/s, turning 1.0 rad/s |
| turn_left | the same, mirrored left-right | | |
| sit | D1_013_KAN01_001 (dog_sit01) | 7.5-11.0 s | stand 1 s, sit down, hold |
| lie | D1_061z_KAN01_003 (dog_walk12) | 35.5-41.0 s | stand, sit, lie down, hold |
