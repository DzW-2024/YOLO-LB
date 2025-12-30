import numpy as np



# Hand-eye calibration transformation matrix
# rotation matrix
ROTATION_MATRIX = np.array([
    [0.999158, -0.039569, 0.010845],
    [0.03992, 0.998616, -0.034238],
    [-0.009476, 0.034642, 0.999355]
])
# Initial pose of the robotic arm
a, b, c = 28.61, 593.34, 635.52

# Translation matrix
TRANSLATION_VECTOR = np.array([-0.0296276 + a, -0.104898 + b, -0.226054 + c])



def camera_to_world(camera_point: np.ndarray) -> np.ndarray:
    """
    3D coordinates of the camera coordinate system → World coordinate system 3D coordinates
    :param camera_point: 3D point in camera coordinate system (cx, cy, cz)
    :return: World coordinate system 3D point (wx, wy, wz)
    """
    # world_point = R * camera_point + t
    world_point = np.dot(ROTATION_MATRIX, camera_point) + TRANSLATION_VECTOR
    return world_point