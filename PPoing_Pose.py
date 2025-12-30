import numpy as np
import math
import torch
from YOLO_LB import *
import pyrealsense2 as rs
import cv2
from Coord_Convert import *

# Keypoint visibility determination threshold
CONFIDENCE_THRESHOLD = 0.9
# Number of candidate points in scenario 5
CANDIDATE_POINTS_NUM = 10
# Scene 5 depth change threshold γth
DEPTH_DIFF_THRESHOLD = 5
# Candidate point expansion step size (Pixels) for scenario 5
STEP_PIXEL = 5
# Detection confidence threshold
CONF_FILTER_THRESHOLD = 0.4


KEYPOINT_NAMES = [
    'main_stem',
    'lateral1',
    'lateral2'
]


def get_aligned_images():
    # Waiting to acquire image frames, acquiring a set of frames for color and depth.
    frames = pipeline.wait_for_frames()
    # Obtain the alignment frame, aligning the depth box with the color box
    aligned_frames = align.process(frames)
    # Retrieve the depth frame from the aligned frame
    aligned_depth_frame = aligned_frames.get_depth_frame()
    # Retrieve the color frame from the aligned frame
    aligned_color_frame = aligned_frames.get_color_frame()

    # Convert images to NumPy arrays
    img_color = np.asanyarray(aligned_color_frame.get_data())  # RGB
    img_depth = np.asanyarray(aligned_depth_frame.get_data())  # Depth

    # Retrieve camera parameters
    # Retrieve depth parameters (used for converting pixel coordinates to camera coordinates)
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # Retrieve camera internal parameters
    color_intrin = aligned_color_frame.profile.as_video_stream_profile().intrinsics

    depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)

    return color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame


def get_3d_camera_coordinate(depth_pixel, aligned_color_frame, aligned_depth_frame, pc):
    x = int(depth_pixel[0])
    y = int(depth_pixel[1])
    # Compute point cloud
    pc.map_to(aligned_color_frame)
    points = pc.calculate(aligned_depth_frame)
    vtx = np.asanyarray(points.get_vertices())
    vtx = np.reshape(vtx, (480, 640, -1))
    camera_coordinate = vtx[y][x][0]
    dis = camera_coordinate[2]
    return dis, camera_coordinate



class KeyPoint:
    """
    Key point data structure:
    Stores 2D/3D coordinates;
    confidence;
    visibility
    """

    def __init__(self, name: str, x2d: float = 0, y2d: float = 0,
                 x3d: float = 0, y3d: float = 0, z3d: float = 0,
                 confidence: float = 0):
        self.name = name  # Keypoint name: main_stem/lateral1/lateral2
        self.x2d, self.y2d = x2d, y2d  # 2D pixel coordinate
        self.x3d, self.y3d, self.z3d = x3d, y3d, z3d  # World coordinate system 3D coordinates
        self.confidence = confidence  # confidence
        self.visible = confidence > CONFIDENCE_THRESHOLD  # visibility determination




def calculate_2d_pruning_pose(lateral1: KeyPoint, lateral2: KeyPoint) -> tuple:
    """
    2D level point localization and pose estimation
    :param lateral1: lateral1 keypoint
    :param lateral2: lateral2 keypoint
    :return: (pruning_point_2d, pose_angle_2d) →  2D pruning point coordinates, 2D pose angles
    """
    # 2D pruning point
    pruning_point_2d = (lateral1.x2d, lateral1.y2d)

    # Calculate the direction vector from lateral1 to lateral2
    dx = lateral2.x2d - lateral1.x2d
    dy = lateral2.y2d - lateral1.y2d
    dir_vector = np.array([dx, dy])

    # Calculate the angle θ2D between the image and the Y-axis
    y_axis = np.array([0, 1])  # Y-axis unit vector in image
    if np.linalg.norm(dir_vector) == 0:
        return pruning_point_2d, 0.0

    # Calculating the angle between vectors using the dot product
    cos_theta = np.dot(dir_vector, y_axis) / (np.linalg.norm(dir_vector) * np.linalg.norm(y_axis))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta2d = np.arccos(cos_theta)
    return pruning_point_2d, theta2d


def process_scenario_1_2(main_stem: KeyPoint, lateral1: KeyPoint, lateral2: KeyPoint,
                         aligned_color_frame, aligned_depth_frame, pc) -> tuple:
    """
    Scenario 1+2：All keypoints visible or main_stem occluded but lateral1/lateral2 visible
    :return: (pruning_point_3d, branch_vector_3d) → 3D pruning points, lateral branch axis vectors
    """
    # Calculate the 3D coordinates of lateral1 and lateral2 in the world coordinate system.
    depth1, cam1 = get_3d_camera_coordinate((lateral1.x2d, lateral1.y2d), aligned_color_frame, aligned_depth_frame, pc)
    depth2, cam2 = get_3d_camera_coordinate((lateral2.x2d, lateral2.y2d), aligned_color_frame, aligned_depth_frame, pc)

    lateral1.x3d, lateral1.y3d, lateral1.z3d = camera_to_world(cam1)
    lateral2.x3d, lateral2.y3d, lateral2.z3d = camera_to_world(cam2)

    # 3D pruning point = 3D coordinates of lateral1
    pruning_point_3d = (lateral1.x3d, lateral1.y3d, lateral1.z3d)
    # lateral axis vector = lateral1→lateral2
    branch_vector = np.array([
        lateral2.x3d - lateral1.x3d,
        lateral2.y3d - lateral1.y3d,
        lateral2.z3d - lateral1.z3d
    ])
    return pruning_point_3d, branch_vector


def process_scenario_3(main_stem: KeyPoint, lateral1: KeyPoint, lateral2: KeyPoint,
                       aligned_color_frame, aligned_depth_frame, pc) -> tuple:
    """
    Scenario：lateral1 occlusion, Use the midpoint between main_stem and lateral2 as the pruning point.
    :return: (pruning_point_3d, branch_vector_3d) → 3D pruning points, lateral branch axis vectors
    """
    # Calculate the 3D coordinates of the main_stem and lateral2 in the world coordinate system.
    depth0, cam0 = get_3d_camera_coordinate((main_stem.x2d, main_stem.y2d),
                                            aligned_color_frame, aligned_depth_frame, pc)
    depth2, cam2 = get_3d_camera_coordinate((lateral2.x2d, lateral2.y2d),
                                            aligned_color_frame, aligned_depth_frame, pc)

    main_stem.x3d, main_stem.y3d, main_stem.z3d = camera_to_world(cam0)
    lateral2.x3d, lateral2.y3d, lateral2.z3d = camera_to_world(cam2)

    # Calculate the 3D coordinates of lateral1 using midpoint interpolation.
    pruning_point_3d = (
        (main_stem.x3d + lateral2.x3d) / 2,
        (main_stem.y3d + lateral2.y3d) / 2,
        (main_stem.z3d + lateral2.z3d) / 2
    )
    # lateral branch axis vector = main_stem→lateral2
    branch_vector = np.array([
        lateral2.x3d - main_stem.x3d,
        lateral2.y3d - main_stem.y3d,
        lateral2.z3d - main_stem.z3d
    ])
    return pruning_point_3d, branch_vector


def process_scenario_4(main_stem: KeyPoint, lateral1: KeyPoint, lateral2: KeyPoint,
                       aligned_color_frame, aligned_depth_frame, pc) -> tuple:
    """
    Scenario：lateral2 occlusion, Replace the lateral branch axis with the vector from main_stem→lateral1
    :return: (pruning_point_3d, branch_vector_3d) → 3D pruning points, lateral branch axis vectors
    """
    # Calculate the 3D coordinates of the world coordinate system for main_stem and lateral1.
    depth0, cam0 = get_3d_camera_coordinate((main_stem.x2d, main_stem.y2d),
                                            aligned_color_frame, aligned_depth_frame, pc)
    depth1, cam1 = get_3d_camera_coordinate((lateral1.x2d, lateral1.y2d),
                                            aligned_color_frame, aligned_depth_frame, pc)

    main_stem.x3d, main_stem.y3d, main_stem.z3d = camera_to_world(cam0)
    lateral1.x3d, lateral1.y3d, lateral1.z3d = camera_to_world(cam1)

    # 3D pruning point = 3D coordinates of lateral1
    pruning_point_3d = (lateral1.x3d, lateral1.y3d, lateral1.z3d)
    # lateral branch axis vector = main_stem→lateral1
    branch_vector = np.array([
        lateral1.x3d - main_stem.x3d,
        lateral1.y3d - main_stem.y3d,
        lateral1.z3d - main_stem.z3d
    ])
    return pruning_point_3d, branch_vector


def process_scenario_5(lateral1: KeyPoint, aligned_color_frame, aligned_depth_frame, pc) -> tuple:
    """
    Scenario5：Both main_stem and lateral2 are occluded; single-keypoint emergency localization
    :return: (pruning_point_3d, branch_vector_3d) → 3D pruning points, lateral branch axis vectors
    """
    # 3D pruning point = 3D coordinates of lateral1
    depth1, cam1 = get_3d_camera_coordinate((lateral1.x2d, lateral1.y2d),
                                            aligned_color_frame, aligned_depth_frame, pc)

    lateral1.x3d, lateral1.y3d, lateral1.z3d = camera_to_world(cam1)
    pruning_point_3d = (lateral1.x3d, lateral1.y3d, lateral1.z3d)

    # Step 1: Expand candidate points along the 2D direction
    x1, y1 = lateral1.x2d, lateral1.y2d
    # The 2D direction vector is lateral1→lateral2
    dir_2d = np.array([0, 1])
    # Standardization
    dir_2d = dir_2d / np.linalg.norm(dir_2d)

    candidate_points_2d = []
    for i in range(1, CANDIDATE_POINTS_NUM + 1):
        x = x1 + dir_2d[0] * i * STEP_PIXEL
        y = y1 + dir_2d[1] * i * STEP_PIXEL
        candidate_points_2d.append((x, y))

    # Step 2: Calculate the depth difference and depth difference variation of candidate points
    valid_candidates_3d = []
    # The depth value of lateral1
    d0 = depth1
    prev_delta_d = 0

    for (x, y) in candidate_points_2d:
        di, cam_candidate = get_3d_camera_coordinate((x, y),
                                                aligned_color_frame, aligned_depth_frame, pc)
        delta_d = abs(di - d0)

        # Calculate the depth difference change γi
        if len(valid_candidates_3d) > 0:
            gamma_i = abs(delta_d - prev_delta_d)
            if gamma_i > DEPTH_DIFF_THRESHOLD:
                # Subsequent points are treated as occlusions/background, ending filtering
                break

        # Add valid candidate points (converted to 3D coordinates in the world coordinate system)
        cam = cam_candidate
        world = camera_to_world(cam)
        valid_candidates_3d.append(world)
        prev_delta_d = delta_d

    # Step 3: Calculate the 3D coordinates of virtual lateral2
    if not valid_candidates_3d:
        virtual_lateral2 = np.array([
            lateral1.x3d,
            lateral1.y3d,
            lateral1.z3d + 0.1
        ])
    else:
        virtual_lateral2 = np.mean(valid_candidates_3d, axis=0)

    # Step 4: Calculate the lateral branch axis vector (lateral1 → virtual lateral2)
    branch_vector = virtual_lateral2 - np.array(pruning_point_3d)
    return pruning_point_3d, branch_vector


def process_occlusion_scenario(key_points: list, pointcloud: np.ndarray,
                               aligned_color_frame, aligned_depth_frame, pc) -> tuple:
    """
   Unify occlusion scene processing, returning 3D pruning points and lateral branch axis vectors
    :param key_points: Keyooints List [main_stem, lateral1, lateral2]
    :param pointcloud: point cloud
    :return: (pruning_point_3d, branch_vector_3d)
    """
    main_stem, lateral1, lateral2 = key_points

    # Scenario determination (based on keypoint visibility)
    if lateral1.visible and lateral2.visible:
        # Scenario 1+2: main_stem visible or invisible
        return process_scenario_1_2(main_stem, lateral1, lateral2, pointcloud, aligned_color_frame, aligned_depth_frame,
                                    pc)
    elif not lateral1.visible and main_stem.visible and lateral2.visible:
        # Scenario 3: Lateral1 occlusion
        return process_scenario_3(main_stem, lateral1, lateral2, aligned_color_frame, aligned_depth_frame, pc)
    elif not lateral2.visible and main_stem.visible and lateral1.visible:
        # Scenario 4: Lateral2 occlusion
        return process_scenario_4(main_stem, lateral1, lateral2, pointcloud, aligned_color_frame, aligned_depth_frame,
                                  pc)
    elif not main_stem.visible and not lateral2.visible and lateral1.visible:
        # Scenario 5: Both main_stem and lateral2 are occluded.
        return process_scenario_5(lateral1, pointcloud, aligned_color_frame, aligned_depth_frame, pc)
    else:
        raise ValueError("Undefined occlusion scenario! Please check the visibility combinations of keypoints.")


def calculate_pitch_angle(branch_vector: np.ndarray) -> float:
    """
    Calculate pitch angle θ
    :param branch_vector: Lateral branch axis vectors (dx, dy, dz)
    :return: Pitch angle
    """
    dx, dy, dz = branch_vector
    # Calculate the magnitude of the projection vector onto the XY plane
    proj_norm = math.hypot(dx, dy)
    if proj_norm == 0:
        return 0.0

    # Pitch angle = arcsin(dz / magnitude of lateral branch axis vector)
    vector_norm = np.linalg.norm(branch_vector)
    pitch = math.asin(dz / vector_norm)
    return pitch


def calculate_yaw_angle(branch_vector: np.ndarray) -> float:
    """
    Calculate the yaw angle θ
    :param branch_vector: lateral branch axis vector (dx, dy, dz)
    :return: yaw angle
    """
    dx, dy, _ = branch_vector
    if dx == 0 and dy == 0:
        return 0.0

    # yaw_angle = arctan2(dy, dx)
    yaw = math.atan2(dy, dx)
    return yaw


from typing import List, Dict, Tuple


def extract_keypoints(result) -> List[Dict]:
    """
    Extract keypoint information
    Args:
        result: Single-frame prediction results
    Returns:
        A list containing keypoint information for each detected object, with each element being a dictionary:
        {
            "person_id": Detection object index,
            "confidence": confidence,
            "keypoints": Keypoints list (Each element contains coordinates, confidence, and name),
            "bbox": Detection box coordinates
        }
    """
    keypoints_list = []
    if result.keypoints is None:
        return keypoints_list

    # Iterate through each detected object
    for LB_idx in range(len(result.keypoints.xy)):
        # Get the bounding box
        bbox = result.boxes.xyxy[LB_idx].cpu().numpy() if result.boxes is not None else None
        # Obtain the confidence of the object
        LB_conf = result.boxes.conf[LB_idx].cpu().numpy() if result.boxes is not None else 0.0

        # Extract all keypoints of this object
        LB_keypoints = []
        keypoints_xy = result.keypoints.xy[LB_idx].cpu().numpy()
        keypoints_conf = result.keypoints.conf[LB_idx].cpu().numpy()  # (3,) Keypoint Confidence

        for kpt_idx in range(len(keypoints_xy)):
            x, y = keypoints_xy[kpt_idx]
            kpt_conf = keypoints_conf[kpt_idx] if len(keypoints_conf) > kpt_idx else 0.0
            kpt_name = KEYPOINT_NAMES[kpt_idx] if kpt_idx < len(KEYPOINT_NAMES) else f"关键点{kpt_idx}"

        LB_keypoints.append({
            "name": kpt_name,
            "x": float(x),
            "y": float(y),
            "confidence": float(kpt_conf)
        })

        # Encapsulate the keypoint information of this object
        keypoints_list.append({
            "LB_id": int(LB_idx),
            "confidence": float(LB_conf),
            "keypoints": LB_keypoints,
            "bbox": bbox.tolist() if bbox is not None else None
        })

    return keypoints_list


if __name__ == "__main__":
    # Camera resolution
    fra = [640, 480]

    pipeline = rs.pipeline()
    config = rs.config()
    # Configure depth and color streams
    # Configure color camera
    config.enable_stream(rs.stream.color, fra[0], fra[1], rs.format.bgr8, 30)
    # Configure the infrared camera
    config.enable_stream(rs.stream.infrared, 1, fra[0], fra[1], rs.format.y8, 30)
    config.enable_stream(rs.stream.infrared, 2, fra[0], fra[1], rs.format.y8, 30)
    # Configure depth image
    config.enable_stream(rs.stream.depth, fra[0], fra[1], rs.format.z16, 30)
    # Start streaming
    profile = pipeline.start(config)

    pc = rs.pointcloud()
    points = rs.points()

    # Create an alignment object. rs.align enables us to align depth frames with other frames.
    # “align_to” specifies the stream type for which the depth frames are to be aligned.
    align_to = rs.stream.color
    align = rs.align(align_to)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    weight_path = 'YOLO_LB.pt'
    model = YOLO_LB(weight_path)
    model.to(device)

    try:
        while True:
            (color_intrin, depth_intrin, img_color, img_depth, depth_mapped_image,
             aligned_color_frame, aligned_depth_frame) = get_aligned_images()

            results = model.predict(source=aligned_color_frame,
                                    imgsz=640,
                                    show=True,
                                    )
            # Extract single-frame results
            result = results[0]
            # Extract keypoint information
            keypoints_info = extract_keypoints(result)

            for LB_info in keypoints_info:
                # LB_info: Information for each bounding box
                print(f"Lateral branch {LB_info['LB_id']} - box confidence：{LB_info['confidence']:.2f}")
                print(f"Bounding box：{LB_info['bbox']}")
                print(f"Keypoint：{len(LB_info['keypoints'])}")
                for kpt in LB_info['keypoints']:
                    # Information for each keypoint within each bounding box
                    print(f"{kpt['name']}: ({kpt['x']:.2f}, {kpt['y']:.2f}) - confidence：{kpt['confidence']:.2f}")

                # Keypoint detection results
                main_stem = KeyPoint(name=kpt[0]['name'], x2d=kpt[0]['x'], y2d=kpt[0]['y'],
                                     confidence=kpt[0]['confidence'])
                lateral1 = KeyPoint(name=kpt[1]['name'], x2d=kpt[1]['x'], y2d=kpt[1]['y'],
                                    confidence=kpt[1]['confidence'])
                lateral2 = KeyPoint(name=kpt[2]['name'], x2d=kpt[2]['x'], y2d=kpt[2]['y'],
                                    confidence=kpt[2]['confidence'])
                key_points = [main_stem, lateral1, lateral2]

                # Simulated point cloud data
                h, w = 480, 640
                pointcloud = np.random.rand(h, w, 3)
                pointcloud[..., 2] = 0.5 + pointcloud[..., 2]

                # 2D localization and pose estimation
                pruning_2d, pose_2d = calculate_2d_pruning_pose(lateral1, lateral2)
                print(f"2D pruning point coordinates：({pruning_2d[0]:.2f}, {pruning_2d[1]:.2f})")
                print(f"2D pose angle：{math.degrees(pose_2d):.2f}°")

                # 3D Localization (Handling occlusion scenarios)
                pruning_3d, branch_vector = process_occlusion_scenario(key_points, pointcloud, aligned_color_frame,
                                                                       aligned_depth_frame, pc)
                print(
                    f"3D pruning point coordinates (WorldCoordinate)：({pruning_3d[0]:.4f}m, {pruning_3d[1]:.4f}m, {pruning_3d[2]:.4f}m)")
                print(
                    f"Lateral branch axial vector：({branch_vector[0]:.4f}, {branch_vector[1]:.4f}, {branch_vector[2]:.4f})")

                # 3D Pose angle calculation
                pitch = calculate_pitch_angle(branch_vector)
                yaw = calculate_yaw_angle(branch_vector)
                print(f"pitch_angle：{math.degrees(pitch):.2f}°")
                print(f"yaw_angle：{math.degrees(yaw):.2f}°")


    finally:
        # Stop streaming
        pipeline.stop()
