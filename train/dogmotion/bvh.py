"""Minimal BVH reader and forward kinematics (for the MANN dog motion capture, see SOURCES.md).

    joints, frames, dt = load(path)          # skeleton + raw channel values
    pos = world_positions(joints, frames)    # (frames, joints, 3) in metres, Z up, X/Y on the floor

Only what the dog files use: one ROOT with 6 channels (position + rotation), every other joint 3
rotation channels, any Euler order given by the CHANNELS line, "End Site" leaves. BVH is Y-up in cm;
positions come back Z-up in metres (x, y, z) = (X, -Z, Y) / 100.
"""
import numpy as np


class Joint:
    def __init__(self, name, parent, offset):
        self.name, self.parent, self.offset = name, parent, np.asarray(offset, float)
        self.channels, self.col = [], 0


def load(path):
    """Parse a BVH file. Returns (joints, frames (n, channels) float array, frame time s)."""
    tokens = open(path).read().split()
    i, joints, stack, col = 0, [], [], 0
    assert tokens[0] == "HIERARCHY"
    i = 1
    while tokens[i] != "MOTION":
        t = tokens[i]
        if t in ("ROOT", "JOINT"):
            joints.append(Joint(tokens[i + 1], stack[-1] if stack else -1, (0, 0, 0)))
            i += 2
        elif t == "End":   # End Site: a leaf joint with no channels
            joints.append(Joint(joints[stack[-1]].name + "_end", stack[-1], (0, 0, 0)))
            i += 2
        elif t == "{":
            stack.append(len(joints) - 1)
            i += 1
        elif t == "}":
            stack.pop()
            i += 1
        elif t == "OFFSET":
            joints[stack[-1]].offset = np.array([float(x) for x in tokens[i + 1:i + 4]])
            i += 4
        elif t == "CHANNELS":
            n = int(tokens[i + 1])
            j = joints[stack[-1]]
            j.channels, j.col = tokens[i + 2:i + 2 + n], col
            col += n
            i += 2 + n
        else:
            raise ValueError(f"unexpected token {t!r}")
    assert tokens[i + 1] == "Frames:" and tokens[i + 3] == "Frame" and tokens[i + 4] == "Time:"
    n, dt = int(tokens[i + 2]), float(tokens[i + 5])
    frames = np.array(tokens[i + 6:i + 6 + n * col], float).reshape(n, col)
    return joints, frames, dt


def _rot(axis, deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    if axis == "X":
        m = [[np.ones_like(a), 0 * a, 0 * a], [0 * a, c, -s], [0 * a, s, c]]
    elif axis == "Y":
        m = [[c, 0 * a, s], [0 * a, np.ones_like(a), 0 * a], [-s, 0 * a, c]]
    else:
        m = [[c, -s, 0 * a], [s, c, 0 * a], [0 * a, 0 * a, np.ones_like(a)]]
    return np.moveaxis(np.array(m), [0, 1], [-2, -1])   # (n, 3, 3)


def world_transforms(joints, frames):
    """Global rotation (n, J, 3, 3) and position (n, J, 3) of every joint, in BVH units/axes."""
    n = len(frames)
    R = np.zeros((n, len(joints), 3, 3))
    P = np.zeros((n, len(joints), 3))
    for k, j in enumerate(joints):
        local = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
        trans = np.broadcast_to(j.offset, (n, 3)).copy()
        for c, name in enumerate(j.channels):
            v = frames[:, j.col + c]
            if name.endswith("position"):
                trans[:, "XYZ".index(name[0])] = v   # the root's position channels replace its offset
            else:
                local = local @ _rot(name[0], v)
        if j.parent < 0:
            R[:, k], P[:, k] = local, trans
        else:
            R[:, k] = R[:, j.parent] @ local
            P[:, k] = P[:, j.parent] + np.einsum("nij,nj->ni", R[:, j.parent], trans)
    return R, P


def to_zup(p):
    """BVH Y-up centimetres -> Z-up metres: (x, y, z) = (X, -Z, Y) / 100."""
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1) / 100.0


def world_positions(joints, frames):
    return to_zup(world_transforms(joints, frames)[1])
