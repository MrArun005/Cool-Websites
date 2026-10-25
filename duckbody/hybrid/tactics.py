"""Team tactics for the 5-a-side duck match (match.py): pure Python (math + numpy), no MuJoCo, so the same decisions
can be ported line by line to JavaScript for the in-browser match.

    tac = TeamTactics(seed)                    # one per team (it is the team's shared brain: roles, pass calls)
    cmds = tac.step(wm)                        # every 50 Hz tick
    cmds["b4"] -> Cmd(vx, vy, wz, head=(yaw, pitch, neck), kick=None | "left" | "right", kind, mate, cam, state, role)

Input, the team's world model `wm` (a dict; positions in metres, field frame, x towards the east goal):
    t           match time (s)
    phase       "playing" | "ready" | "set" | anything else (= stand still, look at the ball)
    dir         +1 if this team attacks +x, -1 if it attacks -x
    ball        the team's shared ball estimate [x, y] (team communication: late, noisy) or None
    setpiece    None or {"kind", "ours": bool, "ball": [x, y], "radius"}: while it is the other team's, keep radius away
    spots       {name: [x, y]}: kick-off spots (READY phase)
    players     list of {"name", "role" ("GK", "D1", "D2", "A1", "A2"), "pos": [x, y], "psi" (heading, rad),
                "active" (not taken off), "down" (fallen / getting up), "busy" (kicking / hopping: commands ignored),
                "ball": own estimate [x, y] or None, "ball_src": "own" (own camera, fresh) | "team" | None,
                "ball_age" (s since that player's own camera saw it)}
    opps        list of {"pos": [x, y], "role", "active"} (opponents as this team perceives them, with noise)

Output per player name: a Cmd. vx, vy, wz go to velstand (`runtime.drive`), head to `runtime.look(yaw, pitch, neck)`,
kick (when not None) starts the ball_kick_left/right brain; `cam` says whether that player should run its head camera
this tick (the game renders only those), `kind` / `mate` label the kick (shot, pass, clearance, forward).

The plan, per tick:
 1. Ball: the striker's own fresh view, else the team estimate. Ball velocity from the last 0.4 s of estimates.
 2. Roles (re-picked every 0.3 s, with hysteresis): GK keeps goal; the STRIKER (lowest cost: distance, turn, fallen,
    pass receiver, last kicker following up) plays the ball; the SECOND (next lowest) presses / covers and joins the
    contest when the ball is within reach; the other two take their role's job: attackers make support runs into
    open space with a clear passing lane, defenders mark the most dangerous opponent goal-side or hold the block.
 3. The block (formation) slides with the ball: up and across. Off-ball players face the ball and move: at their
    spot they jockey (a slow sideways shuffle), support runs re-target every ~2.5 s into new space.
 4. Ball play (soccer.py's kick cycle, faster): approach behind the ball on the line to the aim, align at a foot,
    head home ~1 s, kick. A quick kick fires from any state when the ball is already in a foot's kick window and the
    aim is roughly ahead. Duel when an opponent is close: walk through the ball towards the aim. Aim: shoot when the
    lane is open from <= SHOOT_D, pass to an open teammate further up (lead pass), else forward / clear.
 5. After a pass the receiver turns to the ball and moves onto its path to meet it; the passer runs forward
    (give-and-go). After any other kick the kicker keeps the ball (follow-up) for a few seconds.
 6. Goalkeeper: on the arc in front of its goal on the line ball -> goal, facing the ball, sliding with it; comes out to
    claim a ball in its box that is close to it (or that only it can reach first).
"""
from __future__ import annotations

import math
import random

import numpy as np

# ---------------------------------------------------------------- field (match.py asserts these are its values)
HX, HY = 2.25, 1.5
GOAL_W = 1.3
PEN_AREA = (1.0, 2.5)
BORDER = 0.5
BALL_R = 0.035
DT = 0.02                                   # 50 Hz

# ---------------------------------------------------------------- the duck's ball skills (soccer.py's numbers)
KICK_X, KICK_Y = 0.08, 0.032                # ball centre in the trunk frame for a straight kick
TOL_X, TOL_Y, TOL_H = 0.02, 0.02, 0.15      # alignment tolerances (m, m, rad)
BEHIND, AROUND = 0.24, 0.28                 # approach point behind the ball; going round it at this radius
READY_S = 1.0                               # head home this long before a lined-up kick (the kick brains are blind)
QUICK_READY_S = 0.8                         # quick kick: head home this long (soccer.py probe: 4/4)
QUICK_X, QUICK_Y, QUICK_H = (-0.03, 0.035), 0.03, 0.45   # quick kick window around (KICK_X, +-KICK_Y), aim error
FOOT_VIEW = (-0.6, 1.1)                     # neck, head pitch: the camera sees the ball at the feet
CAM_H, CAM_DOWN = 0.21, 0.17                # camera height; the eye looks ~10 deg down at head pitch 0

# ---------------------------------------------------------------- tactics
DUEL_D = 0.35               # an opponent this close to the ball: no time to line up -> duel (walk through it)
ALIGN_MAX_S = 3.0           # cannot line up in this long -> duel / dribble
SHOOT_D = 2.4               # shoot from up to this far when the lane is open (was 2.0, any lane)
AIM_Y = 0.35                # aim this far from the goal centre (posts at 0.65): the side away from the keeper
DRIBBLE_D = (1.3, 3.2)      # carry the ball (walk through it) towards goal from this far until this close ...
DRIBBLE_FREE = 0.55         # ... while no opponent is this close to the ball
DRIBBLE_MAX_S = 5.0
LANE_R = 0.18               # an opponent this close to the ball's path blocks it
PASS_MIN, PASS_MAX = 0.5, 2.2
PASS_LEAD = 0.2             # pass this far in front of the receiver (towards the goal)
RECEIVE_S = 4.0             # a receiver waits for the pass this long
FOLLOW_S = 3.0              # the kicker keeps the ball (striker bonus) this long after a kick
ROLE_EVERY = 15             # ticks between role picks (0.3 s)
SUPPORT_EVERY = 2.5         # support runs re-target this often (s)
COVER_D = 0.6               # second player: this far goal-side of the ball (outside the duel radius: it closes
                            # the lane, the striker can still line up)
CONTEST_D = 0.45            # second player joins the contest when the ball is this close to it and an opponent is on it
MARK_D = 0.3                # markers stay this far goal-side of their opponent
SHUFFLE_A, SHUFFLE_T = 0.12, 5.0   # jockeying at a spot: +-12 cm sideways, period 5 s
GAP = 0.4                   # off-ball targets keep this far from teammates
CROWD_VX, CROWD_WZ = 0.6, 0.9    # speed limits next to other ducks (turning: velstand 0.75 -> ~0.7 rad/s, 0.9 -> ~0.9)
CROWD_VX_BALL = 0.75        # ... for the player fighting for the ball (it has to push)
PINNED_S = 1.5              # duel: the ball hasn't moved 5 cm in this long -> step round it, try from the side
# ball at the feet, out of the camera's view (under the body, between / behind the feet): the player's estimate is its
# memory ("mem": last fix + odometry + the ball's last velocity), a leg touch ("touch") or a teammate's ("team")
FIND_D = 0.3                # within this: find it at the feet (look down, step back / quarter turn), never a search turn
FIND_BACK_X, FIND_BACK_Y = 0.06, 0.10   # body frame: the ball is here (under / behind the toes) -> step back
                            # (the head camera, head down, sees x 0.04-0.22 m ahead of the trunk, |y| < 0.1 m; head at
                            # home nothing closer than 0.34 m; head pitch 0.6: 0.14 m)
FIND_BACK_VX = -0.45        # velstand ~ -0.1 m/s: 5-10 cm in 0.5-1 s
FIND_MAX_S = 1.5            # still not in view after this long: play it blind (walk through the estimate) for 2 s
# kick-ready approach ("glide") and kick on the move (kick_move_probe.py): the kick brains do nothing unless the head
# is at home (head pitch 0.3 already: 0/12), so the head goes home GLIDE_D before the ball and the duck walks in on
# its memory / the last fixes; the kick starts from a slow walk (drive 0.4) when the kicking foot is in the air
# (probe: 13/16 hits, ~0.8 m/s, no falls; from drive 0.7-1.0 the duck falls in 10-75 % of the kicks)
GLIDE_D = 0.45              # start the glide (head home) this far from the ball, behind it on the aim line
GLIDE_PHI = 0.35            # ... within this angle of the line behind the ball
GLIDE_VX = 0.4              # walk-in command (velstand ~0.2 m/s)
GLIDE_HEAD_X = 0.24         # head pitched down until the ball is this close (body x), then home for the kick
HOME_S = 0.3                # head command at home this long before a kick (probe: from the foot view, pitch 0.3 or
                            # 0.6, 0.25 s after the head command is home -> 18/18; the slew itself is 2 rad/s)
GLIDE_WIN = (-0.025, 0.015)  # kick when the ball is this far from the kick spot (KICK_X) ...
GLIDE_TOL_Y = 0.022         # ... and this close sideways
FOOT_Y = 0.042              # the feet's line: a ball walked through on it goes about straight (probe)
STRIKE_OPP = 0.45           # behind the ball and an opponent this close: no time to glide -> walk through it fast
SCRAMBLE_D = 0.25          # duel: a ball this close but not in front is pushed out from where it is (scramble)
POKE_D = 0.3                # the second player this close to the ball (pressing) plays it (walk through) instead of standing


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def unit(v):
    v = np.asarray(v, float)
    n = float(np.hypot(v[0], v[1]))
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def seg_dist(a, b, p):
    """Distance of p from segment a-b and the position along it (0 at a, 1 at b)."""
    ab = b - a
    L2 = float(ab @ ab)
    s = float(np.clip((p - a) @ ab / L2, 0, 1)) if L2 > 1e-9 else 0.0
    return float(np.linalg.norm(a + s * ab - p)), s


class Cmd:
    __slots__ = ("vx", "vy", "wz", "head", "kick", "kind", "mate", "cam", "state", "role")

    def __init__(self, vx=0.0, vy=0.0, wz=0.0, head=None, kick=None, kind=None, mate=None, cam=False, state="", role=""):
        self.vx, self.vy, self.wz, self.head = vx, vy, wz, head
        self.kick, self.kind, self.mate, self.cam, self.state, self.role = kick, kind, mate, cam, state, role

    def __repr__(self):
        return (f"Cmd({self.role}/{self.state} v=({self.vx:.2f},{self.vy:.2f},{self.wz:.2f}) kick={self.kick} "
                f"{self.kind or ''} cam={self.cam})")


class Mem:
    """What one player remembers between ticks (its ball-play state machine, the reflex, its support run)."""

    def __init__(self, name, rng):
        self.name, self.rng = name, rng
        self.reset(0.0)
        self.phase = rng.uniform(0, 2 * math.pi)       # shuffle phase (players don't jockey in step)

    def reset(self, t):
        self.state, self.state_t = "search", t
        self.aim, self.aim_kind, self.aim_mate = None, None, None
        self.foot, self.ok_t, self.kick_at = None, 0.0, None
        self.search_dir = 1.0
        self.turn_hist, self.backup_until = [], -1.0
        self.duel_hist, self.detour_until, self.detour_sign = [], -1.0, 1.0
        self.run_target, self.run_t = None, -99.0
        self.dribble_block = -1.0
        self.find_block = -1.0

    def set_state(self, s, t):
        if s != self.state:
            self.state, self.state_t = s, t


class Me:
    """A player's view of itself for one tick (pose + its own ball estimate)."""

    def __init__(self, d):
        self.d = d
        self.name, self.role = d["name"], d["role"]
        self.pos = np.asarray(d["pos"], float)
        self.psi = float(d["psi"])

    def to_body(self, p):
        e = np.asarray(p, float) - self.pos
        c, s = math.cos(self.psi), math.sin(self.psi)
        return np.array([c * e[0] + s * e[1], -s * e[0] + c * e[1]])


# ---------------------------------------------------------------- motion primitives
def go_to(me, target, vmax=0.9, face=None, strafe_r=0.0):
    """Turn towards the target and walk when facing it (velstand's dead band: |vx| >= 0.3, |vy| >= 0.4, |wz| >= 0.6).
    With `face` and the target closer than strafe_r: keep facing `face` and step to it forwards, backwards or
    sideways (how an off-ball player shadows the ball). Returns (vx, vy, wz, distance)."""
    e = me.to_body(target)
    d, b = float(np.hypot(*e)), math.atan2(e[1], e[0])
    if face is not None and d < max(strafe_r, 0.08):
        eh = wrap(face - me.psi)
        wz = 0.0 if abs(eh) < 0.15 else math.copysign(float(np.clip(1.4 * abs(eh), 0.6, 1.0)), eh)
        if d < 0.05 or (strafe_r == 0.0 and d < 0.08):
            return 0.0, 0.0, wz, d
        vx = 0.45 if e[0] > 0.03 else (-0.35 if e[0] < -0.03 else 0.0)
        vy = math.copysign(0.45, e[1]) if abs(e[1]) > 0.03 else 0.0
        return vx, vy, wz, d
    wz = float(np.clip(1.6 * b, -1.0, 1.0))
    if abs(b) > 0.12 and abs(wz) < 0.6:
        wz = math.copysign(0.6, b)
    vx = 0.0
    if abs(b) < 0.6 and d > 0.04:
        vx = float(np.clip(0.3 + 1.2 * d, 0.32, vmax))
    if abs(b) > 0.9 and d < 0.3:
        vx = 0.0
    if abs(b) > 2.4 and d < 0.25:                # just behind: back up instead of turning round
        return -0.4, 0.0, 0.0, d
    return vx, 0.0, wz, d


def look_at(me, p):
    """Head command (yaw, pitch, neck) to keep p (on the floor) in view: the foot view when it is at the feet."""
    rel = me.to_body(p)
    d = float(np.hypot(*rel))
    bear = math.atan2(rel[1], rel[0])
    if d < 0.24:
        return (float(np.clip(bear, -0.8, 0.8)), FOOT_VIEW[1], FOOT_VIEW[0])
    elev = math.atan2(CAM_H - BALL_R, max(d - 0.04, 0.05))
    return (bear, elev - CAM_DOWN, None)


def heading_to(a, b):
    v = np.asarray(b, float) - np.asarray(a, float)
    return math.atan2(v[1], v[0])


def clip_field(p, mx=0.15, my=0.15):
    return np.array([float(np.clip(p[0], -HX + mx, HX - mx)), float(np.clip(p[1], -HY + my, HY - my))])


# ---------------------------------------------------------------- the team brain
class TeamTactics:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.mem = {}
        self.striker, self.second = None, None
        self.k = 0
        self.ball_hist = []                    # (t, ball) for the velocity estimate
        self.pass_to = None                    # (receiver name, passer name, t)
        self.last_kick = None                  # (kicker name, kind, t)
        self.prev_phase = None

    # ------------------------------------------------------------ helpers
    def m(self, name):
        if name not in self.mem:
            self.mem[name] = Mem(name, random.Random(self.rng.random()))
        return self.mem[name]

    @staticmethod
    def own_goal(d):
        return np.array([-d * HX, 0.0])

    def opp_pos(self, wm, gk=True):
        return [np.asarray(o["pos"], float) for o in wm["opps"] if o["active"] and (gk or o["role"] != "GK")]

    def nearest_opp(self, wm, p):
        qs = self.opp_pos(wm)
        return min((float(np.linalg.norm(q - p)) for q in qs), default=9.0)

    def ball_vel(self):
        h = self.ball_hist
        if len(h) < 2 or h[-1][0] - h[0][0] < 0.2:
            return np.zeros(2)
        return (h[-1][1] - h[0][1]) / (h[-1][0] - h[0][0])

    # ------------------------------------------------------------ the tick
    def step(self, wm):
        t, d = wm["t"], wm["dir"]
        self._wm = wm
        self.k += 1
        players = [p for p in wm["players"]]
        out = {}
        if wm["phase"] != self.prev_phase:
            for p in players:
                self.m(p["name"]).reset(t)
            self.pass_to, self.last_kick, self.ball_hist = None, None, []
            self.prev_phase = wm["phase"]
        # 1. the ball
        b = wm["ball"]
        for p in players:
            if p["name"] == self.striker and p["ball_src"] == "own":
                b = p["ball"]
        b = None if b is None else np.asarray(b, float)
        if b is not None:
            self.ball_hist = [h for h in self.ball_hist if t - h[0] <= 0.4] + [(t, b.copy())]
        bv = self.ball_vel()
        if wm["phase"] != "playing":
            for p in players:
                out[p["name"]] = self.stopped(wm, p, b)
            return out
        sp = wm["setpiece"]
        restricted = sp is not None and not sp["ours"]
        # 2. roles
        field = [p for p in players if p["role"] != "GK" and p["active"]]
        if self.pass_to is not None and (t - self.pass_to[2] > RECEIVE_S or
                                         not any(p["name"] == self.pass_to[0] and p["active"] for p in players)):
            self.pass_to = None
        if b is not None and field and (self.k % ROLE_EVERY == 1 or self.striker not in [p["name"] for p in field]):
            costs = sorted(field, key=lambda p: self.cost(p, b, d, t))
            self.striker = costs[0]["name"]
            self.second = costs[1]["name"] if len(costs) > 1 else None
        for p in players:
            me = Me(p)
            mem = self.m(me.name)
            if not p["active"]:
                out[me.name] = Cmd(head=look_at(me, b) if b is not None else None, state="off", role="off")
                continue
            if p["down"]:
                mem.set_state("get up", t)
                out[me.name] = Cmd(state="get up", role=self.role_of(me.name, p))
                continue
            if p["busy"]:
                out[me.name] = Cmd(state=mem.state, role=self.role_of(me.name, p))
                continue
            if mem.state in ("get up", "kick"):
                mem.set_state("approach", t)
            role = self.role_of(me.name, p)
            if b is None:
                c = self.no_ball(wm, me, mem, p)
            elif mem.state == "ready" and not restricted:          # a kick under way is finished, whatever the role
                c = self.ball_play(wm, me, mem, p, b)
            elif role == "GK":
                c = self.keeper(wm, me, mem, p, b, bv, restricted, sp)
            elif role == "striker":
                if restricted:
                    u = unit(self.own_goal(d) - np.asarray(sp["ball"], float))
                    tgt = np.asarray(sp["ball"], float) + u * (sp["radius"] + 0.12)
                    c = self.shadow(me, mem, t, tgt, b, strafe=0.4)
                elif self.pass_to is not None and self.pass_to[0] == me.name and float(np.hypot(*bv)) > 0.2 \
                        and float(np.linalg.norm(b - me.pos)) > 0.35:
                    c = self.receive(me, mem, t, b, bv)
                else:
                    c = self.ball_play(wm, me, mem, p, b)
            elif self.last_kick is not None and self.last_kick[0] == me.name and self.last_kick[1] == "pass" \
                    and t - self.last_kick[2] < FOLLOW_S:
                c = self.support_run(wm, me, mem, b, give_and_go=True)
            elif role == "second":
                c = self.second_play(wm, me, mem, p, b, restricted, sp)
            elif p["role"].startswith("A"):
                c = self.support_run(wm, me, mem, b)
            else:
                c = self.defend(wm, me, mem, p, b, restricted, sp)
            c.role = role
            if restricted and role not in ("striker",):
                c = self.respect(me, mem, c, sp, b)
            c = self.give_way(wm, me, c, role)
            c = self.caution(wm, me, c, carrier=role == "striker" or c.state.startswith("contest"))
            out[me.name] = self.reflex(me, mem, t, c)
            if c.kick:
                self.last_kick = (me.name, c.kind, t)
                if c.kind == "pass" and c.mate is not None:
                    self.pass_to = (c.mate, me.name, t)
                    self.striker = c.mate
        return out

    def role_of(self, name, p):
        if p["role"] == "GK":
            return "GK"
        if name == self.striker:
            return "striker"
        if name == self.second:
            return "second"
        return "support" if p["role"].startswith("A") else "defend"

    def cost(self, p, b, d, t):
        """Who goes for the ball: ~seconds to get there (0.3 m/s, turning 1.2 rad/s) + penalties / bonuses."""
        me = Me(p)
        e = me.to_body(b)
        c = float(np.hypot(*e)) / 0.3 + abs(math.atan2(e[1], e[0])) / 1.2
        if p["down"]:
            c += 8.0
        if p["role"].startswith("D") and b[0] * d > 0.6:
            c += 2.5
        if p["role"].startswith("A") and b[0] * d < -1.2:
            c += 1.5
        if p["name"] == self.striker:
            c -= 1.5
        if self.pass_to is not None and self.pass_to[0] == p["name"]:
            c -= 6.0
        if self.last_kick is not None and self.last_kick[0] == p["name"] and self.last_kick[1] != "pass" \
                and t - self.last_kick[2] < FOLLOW_S:
            c -= 2.0
        return c

    # ------------------------------------------------------------ outside play
    def stopped(self, wm, p, b):
        me, mem = Me(p), self.m(p["name"])
        head = look_at(me, b) if b is not None else None
        if wm["phase"] == "ready" and p["active"] and not p["down"] and not p["busy"]:
            s = np.asarray(wm["spots"][me.name], float)
            face = 0.0 if wm["dir"] > 0 else math.pi
            vx, vy, wz, _ = go_to(me, s, 1.0, face=face, strafe_r=0.25)
            return self.reflex(me, mem, wm["t"], Cmd(vx, vy, wz, head=head, state="position", role="ready"))
        return Cmd(head=head, state="wait", role="wait")

    def no_ball(self, wm, me, mem, p):
        """Nobody knows where the ball is: the striker and the GK look round, the others hold their spot."""
        t = wm["t"]
        if me.name == self.striker or p["role"] == "GK":
            return self.search(me, mem, t, cam=True)
        s = self.formation(p["role"], np.zeros(2), wm["dir"], t, mem)
        vx, vy, wz, _ = go_to(me, s, 0.8, face=heading_to(s, np.zeros(2)), strafe_r=0.35)
        ph = t - mem.state_t
        return Cmd(vx, vy, wz, head=(0.9 * math.sin(0.8 * ph), 0.1, None), state="hold")

    def search(self, me, mem, t, cam=True):
        mem.set_state("search", t)
        ph = t - mem.state_t
        nod = 0.5 + 0.5 * math.sin(2 * math.pi * 0.5 * ph)
        return Cmd(0.0, 0.0, 0.9 * mem.search_dir, head=(0.6 * mem.search_dir, 0.1 + 0.8 * nod, -0.3 * nod),
                   cam=cam, state="search")

    def give_way(self, wm, me, c, role):
        """Don't walk into a teammate: one in the corridor ahead (0.35 m, +-0.2 m) -> stop going forward and sidestep
        away from him. The striker (and a player about to kick) has right of way over everyone, the keeper over the
        rest."""
        if c.kick is not None or c.vx <= 0.0:
            return c
        rank = {"striker": 0, "GK": 1}
        for x in wm["players"]:
            if x["name"] == me.name or not x["active"]:
                continue
            if rank.get(role, 2) < rank.get(self.role_of(x["name"], x), 2):
                continue
            rel = me.to_body(x["pos"])
            if 0.0 < rel[0] < 0.35 and abs(rel[1]) < 0.2:
                c.vx = 0.0
                c.vy = -math.copysign(0.45, rel[1])
                c.state += "/give way"
                break
        return c

    def caution(self, wm, me, c, carrier=False):
        """Into a crowd at walking pace: another duck (either team) within 0.4 m ahead -> vx <= CROWD_VX; any within
        0.3 m -> turn at most CROWD_WZ (hard contacts at full speed and fast spins in contact knock ducks over)."""
        if c.kick is not None:
            return c
        others = [np.asarray(x["pos"], float) for x in wm["players"] if x["name"] != me.name and x["active"]]
        others += self.opp_pos(wm)
        for q in others:
            rel = me.to_body(q)
            d = float(np.hypot(*rel))
            if d < 0.4 and rel[0] > 0.0 and abs(rel[1]) < 0.25:
                c.vx = min(c.vx, CROWD_VX_BALL if carrier else CROWD_VX)
            if d < 0.3:
                c.wz = float(np.clip(c.wz, -CROWD_WZ, CROWD_WZ))
        return c

    def spread(self, wm, me, target, r=GAP):
        """Push a target away from teammates closer than r to it (no two players on one spot)."""
        t = np.asarray(target, float).copy()
        for x in wm["players"]:
            if x["name"] == me.name or not x["active"]:
                continue
            v = t - np.asarray(x["pos"], float)
            dd = float(np.hypot(*v))
            if dd < r:
                t = t + unit(v if dd > 1e-6 else me.pos - t) * (r - dd)
        return t

    def reflex(self, me, mem, t, c):
        """Turning on the spot 1.5 s without the heading moving (blocked by a duck or a post): back up 1 s."""
        turning = abs(c.wz) >= 0.5 and c.vx <= 0.0 and c.kick is None
        mem.turn_hist = (mem.turn_hist + [(turning, me.psi)])[-75:]
        if len(mem.turn_hist) == 75 and all(a for a, _ in mem.turn_hist) and \
                abs(wrap(mem.turn_hist[-1][1] - mem.turn_hist[0][1])) < 0.25:
            mem.backup_until, mem.turn_hist = t + 1.0, []
        if t < mem.backup_until and c.kick is None:
            c.vx, c.vy, c.wz = -0.5, 0.0, 0.0
        return c

    def respect(self, me, mem, c, sp, b):
        """The other team's set piece: keep out of the radius around the ball."""
        bb = np.asarray(sp["ball"], float)
        v = me.pos - bb
        if float(np.linalg.norm(v)) < sp["radius"] + 0.05:
            tgt = bb + unit(v) * (sp["radius"] + 0.15)
            vx, vy, wz, _ = go_to(me, tgt, 0.9)
            c.vx, c.vy, c.wz, c.kick = vx, vy, wz, None
        return c

    def shadow(self, me, mem, t, target, b, vmax=0.9, strafe=0.4, cam=False, state="position"):
        """Go to target; near it keep facing the ball and step (forward / back / sideways)."""
        target = clip_field(self.spread(self._wm, me, target), 0.12, 0.12)
        vx, vy, wz, _ = go_to(me, target, vmax, face=heading_to(target, b), strafe_r=strafe)
        mem.set_state(state, t)
        return Cmd(vx, vy, wz, head=look_at(me, b), cam=cam, state=state)

    # ------------------------------------------------------------ formation and off-ball play
    def formation(self, role, b, d, t, mem):
        """Role spot in the field frame: the block slides with the ball (up and across), plus a slow jockey."""
        bt = np.asarray(b, float) * d                      # team frame: own goal at -x
        sh = SHUFFLE_A * math.sin(2 * math.pi * t / SHUFFLE_T + mem.phase)
        if role == "D1":
            s = (np.clip(0.45 * bt[0] - 1.2, -HX + 0.45, 0.2), np.clip(0.35 * bt[1] + 0.45 + sh, -HY + 0.3, HY - 0.3))
        elif role == "D2":
            s = (np.clip(0.45 * bt[0] - 1.2, -HX + 0.45, 0.2), np.clip(0.35 * bt[1] - 0.45 + sh, -HY + 0.3, HY - 0.3))
        elif role == "A1":
            s = (np.clip(bt[0] + 0.4, -0.6, HX - 0.5), np.clip(0.4 * bt[1] + 0.7 + sh, -HY + 0.25, HY - 0.25))
        elif role == "A2":
            s = (np.clip(bt[0] + 0.4, -0.6, HX - 0.5), np.clip(0.4 * bt[1] - 0.7 + sh, -HY + 0.25, HY - 0.25))
        else:
            s = (-HX + 0.3, 0.0)
        return np.array([float(s[0]), float(s[1])]) * d

    def mates_pos(self, wm, me):
        return [np.asarray(p["pos"], float) for p in wm["players"] if p["name"] != me.name and p["active"]]

    def support_run(self, wm, me, mem, b, give_and_go=False):
        """Attacker off the ball: run into open space ahead of / beside the ball with a clear passing lane; a new
        target every SUPPORT_EVERY s (or when reached), never the spot it is standing on."""
        t, d = wm["t"], wm["dir"]
        opps = self.opp_pos(wm)
        mates = self.mates_pos(wm, me)
        if mem.run_target is None or t - mem.run_t > SUPPORT_EVERY or float(np.linalg.norm(mem.run_target - me.pos)) < 0.12:
            best, bs = None, -1e9
            for dist in (0.7, 1.0, 1.3):
                for ang in (-1.2, -0.75, -0.35, 0.0, 0.35, 0.75, 1.2):
                    a = (0.0 if d > 0 else math.pi) + ang
                    c = clip_field(b + dist * np.array([math.cos(a), math.sin(a)]), 0.3, 0.25)
                    fwd = (c[0] - b[0]) * d
                    open_ = min([float(np.linalg.norm(q - c)) for q in opps] + [1.0])
                    lane = min([seg_dist(b, c, q)[0] for q in opps] + [1.0])
                    crowd = min([float(np.linalg.norm(q - c)) for q in mates] + [1.0])
                    goal_d = float(np.linalg.norm(np.array([d * HX, 0.0]) - c))
                    s = 0.6 * fwd + 1.2 * min(open_, 0.8) + 1.0 * min(lane, 0.4) + 0.8 * min(crowd, 0.6) - 0.2 * goal_d
                    s -= 0.6 * float(np.linalg.norm(c - me.pos))          # reachable
                    if float(np.linalg.norm(c - me.pos)) < 0.25:           # move: not where it already is
                        s -= 1.0
                    if give_and_go:
                        s += 0.8 * fwd
                    s += mem.rng.uniform(0, 0.15)
                    if s > bs:
                        best, bs = c, s
            mem.run_target, mem.run_t = best, t
        st = "give-and-go" if give_and_go else "support"
        return self.shadow(me, mem, t, mem.run_target, b, vmax=1.0, strafe=0.3, state=st)

    def defend(self, wm, me, mem, p, b, restricted, sp):
        """Defender off the ball: mark the most dangerous free opponent goal-side; none -> hold the sliding block."""
        t, d = wm["t"], wm["dir"]
        own = self.own_goal(d)
        threats = []
        for q in self.opp_pos(wm, gk=False):
            if q[0] * d < 0.4 and float(np.linalg.norm(q - b)) > 0.3:      # in / near our half, not on the ball
                threats.append(q)
        # each defender takes the threat nearest its own formation spot (D1 / D2 split by side)
        spot = self.formation(p["role"], b, d, t, mem)
        if threats:
            others = [np.asarray(x["pos"], float) for x in wm["players"] if x["role"].startswith("D") and x["name"] != me.name]
            q = min(threats, key=lambda q: float(np.linalg.norm(q - spot)) + float(np.linalg.norm(q - own)) * 0.3)
            if others and len(threats) > 1:
                o = others[0]
                if float(np.linalg.norm(o - q)) < float(np.linalg.norm(me.pos - q)) - 0.2:
                    q = min([x for x in threats if x is not q], key=lambda x: float(np.linalg.norm(x - spot)))
            tgt = q + unit(own - q) * MARK_D + unit(b - q) * 0.1        # goal-side, a bit towards the ball
            return self.shadow(me, mem, t, tgt, b, vmax=1.0, strafe=0.35, state="mark")
        return self.shadow(me, mem, t, spot, b, vmax=0.9, strafe=0.4, state="block")

    def second_play(self, wm, me, mem, p, b, restricted, sp):
        """Second-closest field player: cover goal-side of the ball (opposite side to the striker), press when the
        opponents have it, and join the contest when the ball is within reach (two on the ball)."""
        t, d = wm["t"], wm["dir"]
        own = self.own_goal(d)
        opp_d = self.nearest_opp(wm, b)
        db = float(np.linalg.norm(b - me.pos))
        st = next((x for x in wm["players"] if x["name"] == self.striker), None)
        st_near = st is not None and float(np.linalg.norm(np.asarray(st["pos"], float) - me.pos)) < 0.3
        st_db = float(np.linalg.norm(np.asarray(st["pos"], float) - b)) if st is not None else 9.0
        poke = not restricted and db < POKE_D and not st_near and st_db > 0.2   # pressing right at the ball: play it
        if poke or (not restricted and db < CONTEST_D and opp_d < min(0.45, st_db) and not st_near):   # 2 on it
            mem_state = mem.state
            if mem_state not in ("approach", "chase", "duel", "align", "ready", "dribble", "glide", "find"):
                mem.set_state("duel", t)
            c = self.ball_play(wm, me, mem, p, b)
            c.state = "contest:" + c.state
            return c
        if mem.state in ("duel", "align", "ready", "approach", "chase", "dribble", "glide", "find"):
            mem.set_state("cover", t)
        side = 1.0
        if st is not None:
            u = unit(own - b)
            n = np.array([-u[1], u[0]])
            side = -1.0 if float((np.asarray(st["pos"], float) - b) @ n) > 0 else 1.0
        else:
            u = unit(own - b)
            n = np.array([-u[1], u[0]])
        if opp_d < 0.8 or b[0] * d < 0.0:                   # they have it / it is in our half: press goal-side
            tgt = b + u * COVER_D + n * side * 0.15
            state = "press"
        else:                                              # we have it up the pitch: close support for the rebound
            tgt = b + u * 0.55 + n * side * 0.45
            state = "cover"
        return self.shadow(me, mem, t, tgt, b, vmax=1.1, strafe=0.3, cam=db < 1.0, state=state)

    def receive(self, me, mem, t, b, bv):
        """The pass is coming: turn to the ball and step onto its path to meet it."""
        mem.set_state("receive", t)
        u = unit(bv)
        s = max(0.0, float((me.pos - b) @ u))
        meet = b + u * max(0.0, s - 0.15)                   # on the path, a little towards the ball
        vx, vy, wz, _ = go_to(me, meet, 0.9, face=heading_to(me.pos, b), strafe_r=0.5)
        return Cmd(vx, vy, wz, head=look_at(me, b), cam=True, state="receive")

    # ------------------------------------------------------------ the goalkeeper
    def keeper(self, wm, me, mem, p, b, bv, restricted, sp):
        d, t = wm["dir"], wm["t"]
        own = self.own_goal(d)
        bt = b * d
        in_box = bt[0] < -HX + PEN_AREA[0] + 0.1 and abs(bt[1]) < PEN_AREA[1] / 2 + 0.1
        my_d = float(np.linalg.norm(b - me.pos))
        opp_d = self.nearest_opp(wm, b)
        mate_d = min([float(np.linalg.norm(np.asarray(x["pos"], float) - b)) for x in wm["players"]
                      if x["name"] != me.name and x["active"] and not x["down"]] + [9.0])
        claim = in_box and (my_d < 0.35 or (my_d < 0.7 and my_d < mate_d) or my_d < min(opp_d, mate_d) - 0.1) \
            and not restricted
        if mem.state in ("approach", "chase", "align", "ready", "duel", "dribble", "glide", "find"):
            if not in_box and bt[0] > -HX + PEN_AREA[0] + 0.35:
                mem.set_state("keeper", t)
            else:
                c = self.ball_play(wm, me, mem, p, b)
                c.cam = True
                return c
        elif claim:
            mem.set_state("approach", t)
            c = self.ball_play(wm, me, mem, p, b)
            c.cam = True
            return c
        mem.set_state("keeper", t)
        # on the arc in front of the goal on the line ball -> goal centre; further out when the ball is far away
        far = float(np.clip((bt[0] + HX) / (2 * HX), 0, 1))
        r = 0.25 + 0.2 * far
        tgt = own + unit(b - own) * r
        tt = tgt * d
        tt[0] = max(tt[0], -HX + 0.18)
        tt[1] = float(np.clip(tt[1], -GOAL_W / 2 + 0.1, GOAL_W / 2 - 0.1))
        tgt = tt * d
        vx, vy, wz, _ = go_to(me, tgt, 0.8, face=heading_to(tgt, b), strafe_r=0.3)
        return Cmd(vx, vy, wz, head=look_at(me, b), cam=True, state="keeper")

    # ------------------------------------------------------------ aim
    def choose_aim(self, wm, me, b):
        """(aim point, kind, mate name): shoot when the lane is open from <= SHOOT_D; pass to an open teammate further
        up when that is better (or the shot is blocked); else forward, or a clearance from our own third."""
        d = wm["dir"]
        gk = next((o for o in wm["opps"] if o["role"] == "GK" and o["active"]), None)
        gy = float(gk["pos"][1]) if gk is not None else 0.0
        side = -math.copysign(1.0, gy) if abs(gy) > 0.08 else math.copysign(1.0, b[1] if abs(b[1]) > 0.05 else 1.0)
        goal = np.array([d * HX, side * AIM_Y])         # the corner away from the keeper (or the near one)
        opps = self.opp_pos(wm, gk=False)
        dg = float(np.linalg.norm(goal - b))

        def lane_open(a, c, r=LANE_R):
            for q in opps:
                dd, s = seg_dist(a, c, q)
                if dd < r and 0.03 < s and float(np.linalg.norm(q - a)) < 1.8:
                    return False
            return True
        shot_open = lane_open(b, goal)
        best, bs = None, -9.0
        for m in wm["players"]:
            if m["name"] == me.name or m["role"] == "GK" or not m["active"] or m["down"]:
                continue
            mp = np.asarray(m["pos"], float)
            fwd, dist = (mp[0] - b[0]) * d, float(np.linalg.norm(mp - b))
            if fwd > 0.15 and PASS_MIN < dist < PASS_MAX and lane_open(b, mp, 0.2):
                free = min([float(np.linalg.norm(q - mp)) for q in opps] + [1.0])
                if free < 0.35:
                    continue
                mg = float(np.linalg.norm(np.array([d * HX, 0.0]) - mp))
                sc = fwd - 0.3 * dist + 0.5 * min(free, 0.8) + (0.6 if mg < dg - 0.6 else 0.0)
                if sc > bs:
                    best, bs = (m, mp, mg), sc
        if shot_open and dg < 1.6:
            return goal, "shot", None
        if best is not None and (not shot_open or dg > SHOOT_D or best[2] < dg - 0.9):
            m, mp, _ = best
            lead = mp + np.array([d * PASS_LEAD, 0.0])
            return lead, "pass", m["name"]
        if dg < SHOOT_D:                               # at goal from in range, open lane or not (best chance left)
            return goal, "shot", None
        if b[0] * d < -0.75 or me.role == "GK":
            return goal, "clearance", None
        return goal, "forward", None

    # ------------------------------------------------------------ ball play
    def find(self, me, mem, t, rel):
        """Ball near the feet but not in view: gaze straight down at where it should be (foot view); under / behind
        the toes -> step back; to the side -> a quarter turn towards it; else hold still for the camera."""
        mem.set_state("find", t)
        bear = math.atan2(rel[1], rel[0])
        head = (float(np.clip(bear, -0.8, 0.8)), FOOT_VIEW[1], FOOT_VIEW[0]) if rel[0] < 0.18 else \
            (float(np.clip(bear, -0.8, 0.8)), 0.6, None)
        if rel[0] < FIND_BACK_X and abs(rel[1]) < FIND_BACK_Y:
            return Cmd(FIND_BACK_VX, 0.0, 0.0, head=head, cam=True, state="find:back")
        if abs(bear) > 1.0:                          # quarter turn towards it, with a sidestep (a turn on the spot is
            return Cmd(0.0, math.copysign(0.45, bear), math.copysign(1.0, bear), head=head, cam=True,   # easily blocked)
                       state="find:turn")
        return Cmd(0.0, 0.0, 0.0, head=head, cam=True, state="find:look")

    def scramble(self, wm, me, rel, b):
        """Contested ball right at the duck but not in front of it (pinned at a foot, beside the body): push it out
        from where it is: in front -> walk through it; beside -> sidestep into it while turning towards it (drags
        it out sideways); never when that pushes it towards our own goal, never for a ball behind. Turning on the
        spot next to another duck mostly fails (it blocks the turn), a sidestep does not."""
        dball = float(np.hypot(*rel))
        if dball > SCRAMBLE_D:
            return None
        bear = math.atan2(rel[1], rel[0])
        if abs(bear) > 1.9:
            return None
        push = (0.0 if abs(bear) < 0.5 else math.copysign(math.pi / 2, bear)) + me.psi    # field direction of the push
        own = self.own_goal(wm["dir"])
        if float(np.array([math.cos(push), math.sin(push)]) @ unit(own - b)) > 0.5:
            return None
        if abs(bear) < 0.5:
            return 0.9, 0.0, float(np.clip(1.6 * bear, -0.7, 0.7))
        return (0.35 if rel[0] > 0.04 else 0.0), math.copysign(0.45, bear), math.copysign(1.0, bear)

    def past_opponent(self, wm, b, k):
        """Walk-through direction: the aim, unless an opponent stands on that line right at the ball (pushing into
        his feet pins it): then 0.8 rad to the side away from him."""
        qs = [q for q in self.opp_pos(wm) if float(np.linalg.norm(q - b)) < 0.4]
        if not qs:
            return k
        q = min(qs, key=lambda q: float(np.linalg.norm(q - b)))
        ka = math.atan2(k[1], k[0])
        ang = wrap(heading_to(b, q) - ka)
        if abs(ang) > 0.7:
            return k
        a = ka - math.copysign(0.8, ang)
        return np.array([math.cos(a), math.sin(a)])

    def pick_foot(self, wm, rel, b):
        foot = "left" if rel[1] > 0 else "right"
        if b[1] > HY - 0.15:
            foot = "left" if wm["dir"] > 0 else "right"
        elif b[1] < -HY + 0.15:
            foot = "right" if wm["dir"] > 0 else "left"
        return foot

    def glide(self, wm, me, mem, p, b, rel, k, n, r, eh, opp_d, dphi):
        """Kick-ready approach: head at home, walk in slowly on the line behind the ball steering the ball onto the
        kicking foot (memory + odometry once the camera no longer sees it), and kick without stopping when it is in
        the window and that foot is in the air. Ball too close -> step back; missed it -> walk through it."""
        t = wm["t"]
        dball = float(np.hypot(*rel))
        # head pitched 0.6 (the camera sees 0.14-0.4 m ahead) until the ball is GLIDE_HEAD_X ahead, then home: the
        # last fixes are close, and the head has been home > HOME_S when the ball reaches the kick window
        home = (0.0, 0.6, None) if rel[0] > GLIDE_HEAD_X else (0.0, 0.0, None)

        def c(vx=0.0, vy=0.0, wz=0.0, state="glide"):
            return Cmd(vx, vy, wz, head=home, cam=True, state=state)
        if dball > GLIDE_D + 0.15 or abs(dphi) > 0.8 or p["ball_src"] not in ("own", "mem"):
            mem.set_state("approach", t)
            return c(state="approach")
        if opp_d < DUEL_D - 0.1:
            mem.set_state("duel", t)
            return c(state="duel")
        ty = KICK_Y if mem.foot == "left" else -KICK_Y
        ex, ey = rel[0] - KICK_X, rel[1] - ty
        if ex < -0.045:                              # too close (under the toes): step back, then kick
            return c(FIND_BACK_VX, 0.0, 0.0, state="glide:back")
        if GLIDE_WIN[0] < ex < GLIDE_WIN[1] and abs(ey) < GLIDE_TOL_Y and abs(eh) < 0.3:
            if p.get("head_home", 0.0) >= HOME_S and (p.get("swing") == mem.foot or p.get("moving", 1.0) < 0.05):
                mem.set_state("kick", t)
                return Cmd(head=home, kick=mem.foot, kind=mem.aim_kind or "forward", mate=mem.aim_mate, cam=True,
                           state="kick")
            if p.get("head_home", 0.0) < HOME_S:        # head not home long enough yet: stop and wait for it
                return c(state="glide:wait")
            return c(GLIDE_VX, 0.0, 0.0, state="glide:phase")   # wrong foot in the air: one more step
        if ex < GLIDE_WIN[0]:                        # just past the window: back off a little
            return c(FIND_BACK_VX, 0.0, 0.0, state="glide:back")
        # steer: heading on the aim, the trunk's sideways offset from the ball's line -> the kicking foot's line
        lat = float(r @ n)
        psi_d = math.atan2(k[1], k[0]) - float(np.clip(4.0 * (lat + ty), -0.35, 0.35))
        e = wrap(psi_d - me.psi)
        wz = 0.0 if abs(e) < 0.08 else float(np.clip(2.0 * e, -0.7, 0.7))
        if ex < 0.06 and abs(ey) > GLIDE_TOL_Y:      # close but off to the side: sidestep onto it
            return c(0.0, math.copysign(0.45, ey), wz, state="glide:side")
        return c(GLIDE_VX, 0.0, wz)

    def ball_play(self, wm, me, mem, p, b):
        """soccer.py's kick cycle with the duel; returns a Cmd (kick set on the tick the kick starts)."""
        t = wm["t"]
        src = p["ball_src"]
        fresh_age = p.get("ball_age", 9.0)
        if p["ball_src"] in ("own", "mem", "touch") and p["ball"] is not None:
            b = np.asarray(p["ball"], float)
        rel = me.to_body(b)
        dball = float(np.hypot(*rel))
        if mem.state in ("search", "position", "keeper", "support", "block", "mark", "press", "cover", "receive",
                         "give-and-go", "hold", "wait", "find"):
            mem.set_state("approach", t)
            mem.search_dir = 1.0 if math.atan2(rel[1], rel[0]) >= 0 else -1.0
        opp_d = self.nearest_opp(wm, b)
        if mem.state in ("approach", "chase", "duel", "dribble") or mem.aim is None:
            mem.aim, mem.aim_kind, mem.aim_mate = self.choose_aim(wm, me, b)
        k = unit(mem.aim - b)
        n = np.array([-k[1], k[0]])
        r = me.pos - b
        along, lat = float(r @ k), float(r @ n)
        phi_d, phi_p = math.atan2(r[1], r[0]), math.atan2(-k[1], -k[0])
        dphi = wrap(phi_p - phi_d)
        lim = np.array([HX + BORDER - 0.12, HY + BORDER - 0.12])
        own_fresh = src == "own" and fresh_age < 0.3
        eh = wrap(math.atan2(k[1], k[0]) - me.psi)

        def cmd(vx=0.0, vy=0.0, wz=0.0, head=None, state=None):
            return Cmd(vx, vy, wz, head=head if head is not None else look_at(me, b), cam=True,
                       state=state or mem.state)

        # the ball is at the feet but the camera has lost it: look down and bring it into view
        if dball < FIND_D and mem.state != "ready" and (src in ("mem", "touch") or (src == "team" and fresh_age > 0.3)) \
                and t > mem.find_block:
            if mem.state == "find" and t - mem.state_t > FIND_MAX_S:
                mem.find_block = t + 2.0
                mem.set_state("duel", t)
            else:
                return self.find(me, mem, t, rel)
        # quick kick: the ball is already in a foot's kick window and the aim roughly ahead -> head up, kick
        # (moving states only when the head is already home: they never stop to wait for it)
        home_ok = p.get("head_home", 0.0) >= HOME_S
        if mem.state in ("approach", "chase", "duel", "align", "dribble") and own_fresh and abs(eh) < QUICK_H \
                and (mem.state == "align" or home_ok) \
                and not (mem.state == "dribble" and float(np.linalg.norm(np.array([wm["dir"] * HX, 0.0]) - b)) > SHOOT_D):
            for foot, ty in (("left", KICK_Y), ("right", -KICK_Y)):
                ex, ey = rel[0] - KICK_X, rel[1] - ty
                if QUICK_X[0] < ex < QUICK_X[1] and abs(ey) < QUICK_Y:
                    mem.foot = foot
                    if home_ok:                      # head already home: kick now, no stop
                        mem.set_state("kick", t)
                        return Cmd(head=(0.0, 0.0, None), kick=foot, kind=mem.aim_kind or "forward", mate=mem.aim_mate,
                                   cam=True, state="kick")
                    mem.kick_at = dict(foot=foot, kind=mem.aim_kind, mate=mem.aim_mate, quick=True)
                    mem.set_state("ready", t)
                    return cmd(head=(0.0, 0.0, None))
        if mem.state in ("approach", "chase") and opp_d < DUEL_D and dball < 0.7:
            mem.set_state("duel", t)
        if mem.state == "duel":
            if opp_d > DUEL_D + 0.2 and t - mem.state_t > 0.6:
                mem.set_state("approach", t)
            nh = int(PINNED_S / DT)
            mem.duel_hist = (mem.duel_hist + [b.copy()])[-nh:]
            if len(mem.duel_hist) == nh and np.linalg.norm(mem.duel_hist[0] - b) < 0.05 and t > mem.detour_until:
                mem.detour_until, mem.detour_sign, mem.duel_hist = t + 1.2, mem.rng.choice([-1.0, 1.0]), []
            if t < mem.detour_until - 0.8:           # pinned: first step back (frees a ball trapped at the feet) ...
                return cmd(-0.5, 0.0, 0.0, state="duel:back")
            if t < mem.detour_until:                 # ... then step round it (60 deg) and try from the side
                a = phi_d + mem.detour_sign * 1.0
                target = np.clip(b + 0.22 * np.array([math.cos(a), math.sin(a)]), -lim, lim)
                vx, vy, wz, _ = go_to(me, target, 1.0)
                return cmd(vx, vy, wz)
            if abs(dphi) > 1.0:                      # not behind it: play it from where it is (step / sidestep into
                sc = self.scramble(wm, me, rel, b)    # it) unless that sends it towards our goal; else tight orbit
                if sc is not None:
                    return cmd(*sc, state="duel:scramble")
                a = phi_d + float(np.clip(dphi, -0.9, 0.9))
                target = np.clip(b + 0.2 * np.array([math.cos(a), math.sin(a)]), -lim, lim)
                vx, vy, wz, _ = go_to(me, target, 1.0)
            else:                                    # behind it: walk through it towards the aim, briskly, the ball
                sg = 1.0 if rel[1] > 0 else -1.0     # on a foot's line (goes about straight; probe: ~0.9 m/s)
                kk = self.past_opponent(wm, b, k)
                nn = np.array([-kk[1], kk[0]])
                vx, vy, wz, _ = go_to(me, b + 0.15 * kk - sg * FOOT_Y * nn, 1.0)
                if vx > 0.0 and abs(math.atan2(rel[1], rel[0])) < 0.5:
                    vx = max(vx, 0.9)
                return cmd(vx, vy, wz, state="duel:strike")
            return cmd(vx, vy, wz)
        if mem.state in ("approach", "chase"):
            mem.set_state("chase" if dball > 0.6 else "approach", t)
            if abs(dphi) > 0.6:
                if abs(dphi) > 2.6:
                    room = [min(HY + 0.3 - abs((b + sg * AROUND * n)[1]), HX + 0.3 - abs((b + sg * AROUND * n)[0]))
                            for sg in (1.0, -1.0)]
                    dphi = abs(dphi) if room[0] >= room[1] else -abs(dphi)
                a = phi_d + float(np.clip(dphi, -0.9, 0.9))
                rad = max(AROUND, min(dball, 0.5))
                target = np.clip(b + rad * np.array([math.cos(a), math.sin(a)]), -lim, lim)
                vx, vy, wz, _ = go_to(me, target, 1.1 if dball > 0.8 else 0.9)
            else:
                dg = float(np.linalg.norm(np.array([wm["dir"] * HX, 0.0]) - b))
                if mem.aim_kind in ("shot", "forward") and DRIBBLE_D[0] < dg < DRIBBLE_D[1] and opp_d > DRIBBLE_FREE \
                        and abs(dphi) < 0.35 and dball < 0.3 and t > mem.dribble_block:
                    mem.set_state("dribble", t)
                    return cmd()
                bspeed = float(np.hypot(*self.ball_vel()))
                if abs(dphi) < GLIDE_PHI and dball < GLIDE_D and src in ("own", "mem") and bspeed < 0.25:
                    if opp_d < STRIKE_OPP:           # pressed: walk through it now (duel = walk-through strike)
                        mem.set_state("duel", t)
                        return cmd()
                    mem.set_state("glide", t)
                    mem.foot = self.pick_foot(wm, rel, b)
                    return cmd(head=(0.0, 0.0, None))
                P = np.clip(b - BEHIND * k, -lim, lim)
                vx, vy, wz, dd = go_to(me, P, 1.1 if dball > 0.8 else 0.9, face=math.atan2(k[1], k[0]))
                if src == "own" and (dd < 0.08 or (along < -0.12 and abs(lat) < 0.07 and dball < BEHIND + 0.08)):
                    mem.set_state("align", t)
                    mem.foot = None
            return cmd(vx, vy, wz)
        if mem.state == "glide":
            return self.glide(wm, me, mem, p, b, rel, k, n, r, eh, opp_d, dphi)
        if mem.state == "dribble":                   # carry it: walk through the ball towards the goal
            goal = np.array([wm["dir"] * HX, 0.0])
            dg = float(np.linalg.norm(goal - b))
            if dg < DRIBBLE_D[0] or opp_d < DUEL_D or t - mem.state_t > DRIBBLE_MAX_S or dball > 0.35 or abs(dphi) > 1.0:
                mem.dribble_block = t + 2.0          # then line up a shot / fight for it / start again
                mem.set_state("approach", t)
                return cmd()
            kd = unit(goal - b)
            nd = np.array([-kd[1], kd[0]])
            sg = 1.0 if rel[1] > 0 else -1.0
            vx, vy, wz, _ = go_to(me, b + 0.15 * kd - sg * FOOT_Y * nd, 0.75)
            return cmd(vx, vy, wz)
        if mem.state == "align":
            head = (0.0, FOOT_VIEW[1], FOOT_VIEW[0])
            if dball > BEHIND + 0.15 or along > 0.0:
                mem.set_state("approach", t)
                return cmd(head=head)
            if opp_d < DUEL_D - 0.1 or t - mem.state_t > ALIGN_MAX_S:
                mem.set_state("duel", t)
                return cmd(head=head)
            if mem.foot is None:
                mem.foot = "left" if rel[1] > 0 else "right"
                if b[1] > HY - 0.15:
                    mem.foot = "left" if wm["dir"] > 0 else "right"
                elif b[1] < -HY + 0.15:
                    mem.foot = "right" if wm["dir"] > 0 else "left"
            ty = KICK_Y if mem.foot == "left" else -KICK_Y
            ex, ey = rel[0] - KICK_X, rel[1] - ty
            fresh = src == "own" and fresh_age < 0.25
            vx = vy = wz = 0.0
            if abs(eh) > 0.3:
                wz = math.copysign(0.65, eh)
                if ex > 0.08:
                    vx = 0.33
            else:
                if abs(ex) > TOL_X:
                    vx = 0.33 if ex > 0 else -0.4
                if abs(ey) > TOL_Y:
                    vy = 0.45 if ey > 0 else -0.45
                if abs(eh) > TOL_H:
                    wz = math.copysign(0.6, eh)
            mem.ok_t = mem.ok_t + DT if (vx == vy == wz == 0.0 and fresh) else 0.0
            if mem.ok_t > 0.2:
                mem.set_state("ready", t)
                mem.ok_t = 0.0
                mem.kick_at = dict(foot=mem.foot, kind=mem.aim_kind, mate=mem.aim_mate)
            return cmd(vx, vy, wz, head=head)
        if mem.state == "ready":
            quick = (mem.kick_at or {}).get("quick")
            ty = KICK_Y if mem.foot == "left" else -KICK_Y
            if src in ("own", "mem", "touch") and (abs(rel[0] - KICK_X) > 0.06 or abs(rel[1] - ty) > 0.05):
                mem.set_state("approach", t)             # the ball moved away while the head came up: start again
                return cmd()
            if t - mem.state_t >= (QUICK_READY_S if quick else READY_S) or \
                    (t - mem.state_t >= 0.1 and p.get("head_home", 0.0) >= HOME_S):
                mem.set_state("kick", t)
                ka = mem.kick_at or {}
                return Cmd(head=(0.0, 0.0, None), kick=mem.foot, kind=ka.get("kind") or "forward", mate=ka.get("mate"),
                           cam=True, state="kick")
            return cmd(head=(0.0, 0.0, None))
        mem.set_state("approach", t)
        return cmd()
