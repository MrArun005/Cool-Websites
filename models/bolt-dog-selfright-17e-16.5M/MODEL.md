# Bolt self-righting brain, run 17e at 16.5M steps (real Unitree A1 motors)

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 33.5 N·m, 420 W, 21 rad/s (today's body). How determined: train/jobs/run17e-selfright-a1.json (27 Oct 11:58, after fd5199b) has no BOLT_MOTOR_TORQUE override, and this card says so. Fine-tuned from 17d-13M, which was trained at 17 N·m.

Rolls Bolt from any fall onto its belly; then the posture brain (bolt-dog-posture-getup-3M) tucks the legs and stands.
Real A1 motor model: 33.5 N·m, 420 W, 21 rad/s per joint; ±48° hip abduction. Fine-tuned from run 17d at 13M.

- Exam (exam.txt): belly-down from side, back and half tipped 100%, in about 0.6-0.7 s.
- Full recovery (self-right, tuck, stand), 5 falls each: side, back and half tipped 15/15; from standing 4/5.
- Published reference: learned recovery 98% in 1.2 s (AFR, Go1, sim); ANYmal over 97% in under 5 s (real).
