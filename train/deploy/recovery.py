"""Fall hand-over for the robot program (PLAN.md, recovery chain): the same code run_go2.py and lockstep_recovery.py use.

    rec = Recovery(ctl)                       # ctl: the Go2Controller driving the motion brain
    tgt = rec.step(quat, gyro, q, dq, v_body, height, resume="trot")
    # None: not recovering, run ctl.step as usual. Else: send tgt (self-right or get-up), the plan clock stays paused.

Upright < 0.5 (tilted past 60 degrees) -> the self-right brain (deploy/selfright.py, numpy) until upright > 0.9 for
0.3 s in a row (training's "righted" also asks the body not to slide, which a real Go2 cannot measure well; the brain
keeps rocking on its belly at 3-7 rad/s, so the gyro is no use as a stillness test) -> the motion brain's getup clip from its start -> back to `resume`. Tipping over again
during the getup goes back to self-right. Gives up (rec.failed set; the caller goes to damping mode) after 4 s of
self-right without righting, or 3 falls in a row without getting back to the command.
Only what a real Go2 measures is used: IMU, joints and the estimator's velocity/height.
"""
import numpy as np

from controller import DT, quat_to_mat
from selfright import DEFAULT, SelfRighter

FALL_UP, RIGHT_UP, STILL_STEPS = 0.5, 0.9, 15
SR_TIMEOUT, MAX_FALLS = 4.0, 3


class Recovery:
    def __init__(self, ctl, selfright=DEFAULT):
        self.ctl, self.sr = ctl, SelfRighter(selfright, body=ctl.body.key)
        self.mode, self.failed, self.falls, self.recovered = None, None, 0, 0
        self.t = self.still = 0
        self.events = []

    def step(self, quat, gyro, q, dq, v_body, height, resume):
        up = quat_to_mat(np.asarray(quat, float))[2, 2]
        if self.mode != "selfright" and up < FALL_UP:
            self.falls += 1
            if self.falls > MAX_FALLS:
                self.failed = f"{MAX_FALLS} falls in a row without getting back up"
                return None
            self.mode, self.t, self.still = "selfright", 0, 0
            self.sr.last_action[:] = 0.0
            self.events.append(("fall", up))
        if self.mode == "selfright":
            self.t += 1
            self.still = self.still + 1 if up > RIGHT_UP else 0
            if self.still >= STILL_STEPS:
                self.mode = "getup"
                self.ctl.last_action[:] = 0.0
                self.ctl.start_at("getup", 0.0)
                self.events.append(("righted", self.t * DT))
            elif self.t * DT > SR_TIMEOUT:
                self.failed = f"not righted after {SR_TIMEOUT:g} s of self-right"
                return None
            else:
                return self.sr.step(quat, gyro, q, dq, v_body, height)
        if self.mode == "getup":
            if self.ctl.tau < self.ctl.clips["getup"].length:
                return self.ctl.step(quat, gyro, q, dq, v_body, height)
            self.mode, self.falls = None, 0
            self.recovered += 1
            self.ctl.command(resume, q=q)
            self.events.append(("resumed", resume))
        return None
