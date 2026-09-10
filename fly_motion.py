"""Projects reservoir activity onto a 2-D motor drive (dx, dy, dtheta) and
integrates simple damped physics from it, so a fly icon visibly jerks
around driven by real neuron activations."""
import numpy as np

DAMPING = 0.7
VEL_GAIN = 0.05
ANGLE_GAIN = 0.3
WALL = 1.0
BOUNCE = -0.5


def load_projection(n_neurons, seed=1):
    rng = np.random.default_rng(seed)
    return (rng.uniform(-1, 1, size=(3, n_neurons)) / np.sqrt(n_neurons)).astype(np.float32)


def step(state, projection, pos, vel, angle):
    """state: (n_neurons,) reservoir activation. pos/vel: (2,) arrays,
    angle: float. Returns the updated (pos, vel, angle)."""
    drive = projection @ state
    vel = DAMPING * vel + VEL_GAIN * drive[:2]
    pos = pos + vel
    for i in range(2):
        if pos[i] > WALL:
            pos[i], vel[i] = WALL, vel[i] * BOUNCE
        elif pos[i] < -WALL:
            pos[i], vel[i] = -WALL, vel[i] * BOUNCE
    angle = angle + ANGLE_GAIN * drive[2]
    return pos, vel, angle
