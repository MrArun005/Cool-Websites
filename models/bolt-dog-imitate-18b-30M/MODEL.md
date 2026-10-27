# Bolt dog motion brain, run 18b best (finished at 30M steps)

The run the owner calls "perfect act". 8-slot layout (walk, trot, pace, canter, turn left/right, sit, lie).

Exam (best.json, tracking): walk 92%, trot 90%, pace 93%, turn left 89%, turn right 90%, sit 95%, lie 93%, canter 54% (no falls).
The earlier saved 18b brain is models/bolt-dog-imitate-18b-17.5M. Load with `ImitatePolicy`.
