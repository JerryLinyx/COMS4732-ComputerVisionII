import math
import numpy as np

def look_at_origin(pos):
  # Camera looks towards the origin
  forward = -pos / np.linalg.norm(pos)  # Normalize the direction vector

  # Define up vector (assuming y-up)
  up = np.array([0, 1, 0])

  # Compute right vector using cross product
  right = np.cross(up, forward)
  right = right / np.linalg.norm(right)

  # Recompute up vector to ensure orthogonality
  up = np.cross(forward, right)

  # Create the camera-to-world matrix
  c2w = np.eye(4)
  c2w[:3, 0] = right
  c2w[:3, 1] = up
  c2w[:3, 2] = forward
  c2w[:3, 3] = pos

  return c2w

def rot_x(phi):
    return np.array([
        [math.cos(phi), -math.sin(phi), 0, 0],
        [math.sin(phi), math.cos(phi), 0, 0],
        [0,0,1,0],
        [0,0,0,1],
    ])

# TODO: Change start position to a good position for your scene such as
# the translation vector of one of your training camera extrinsics
START_POS = np.array([1., 0., 0.])
NUM_SAMPLES = 60

frames = []
for phi in np.linspace(360., 0., NUM_SAMPLES, endpoint=False):
    c2w = look_at_origin(START_POS)
    extrinsic = rot_x(phi/180.*np.pi) @ c2w

    # Generate view for this camera pose
    # TODO: Add code for generating a view with your model from the current extrinsic
    frame = ...
    frames.append(frame)
