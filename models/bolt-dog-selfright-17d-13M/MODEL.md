# Bolt self-righting brain, run 17d at 13M steps

> **Motor torque (note 28 Oct; the body is now 33.5 N·m):** trained and examined with 17 N·m with the 420 W / 21 rad/s limits (NOT today's 33.5 N·m body). How determined: train/jobs/run17d-selfright.json sets BOLT_MOTOR_TORQUE=17 (kept after fd5199b, see its message), the run resumed under the power/speed limits of df69a28, and this card says so. Examine with BOLT_MOTOR_TORQUE=17 to reproduce.

Rolls Bolt from any fall onto its belly, then the posture brain (bolt-dog-posture-getup-3M) tucks the legs and stands.
Trained with joints capped at 17 N·m (420 W, 21 rad/s), ±48° hip abduction, full joint range.

- Exam (exam.txt): belly-down from side, back and half tipped 100%, in about 0.7 s.
- Full recovery (self-right, tuck, stand), 5 falls each: side 5/5, back 5/5, half tipped 5/5; from standing 4/5.
- Chosen over the final 15M checkpoint, whose full recovery from half tipped dropped to 2/5.
