import cv2
import numpy as np
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# 6-point eye contours used for EAR, in mediapipe's 478-point face mesh indexing
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# iris landmarks (only present when refine_landmarks=True)
# note: 468-472 sits inside the 33/133 eye contour, and 473-477 inside 362/263
RIGHT_IRIS = [468, 469, 470, 471, 472]
LEFT_IRIS = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)

# 3D reference points (generic face model, in mm) for solvePnP head pose
MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),        # nose tip (1)
    (0.0, -63.6, -12.5),    # chin (152)
    (-43.3, 32.7, -26.0),   # left eye left corner (33)
    (43.3, 32.7, -26.0),    # right eye right corner (263)
    (-28.9, -28.9, -24.1),  # left mouth corner (61)
    (28.9, -28.9, -24.1),   # right mouth corner (291)
], dtype=np.float64)
POSE_LANDMARK_IDS = [1, 152, 33, 263, 61, 291]


def _dist(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))


def _eye_aspect_ratio(pts):
    p1, p2, p3, p4, p5, p6 = pts
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    return vertical / (2.0 * horizontal) if horizontal > 0 else 0.0


def _gaze_ratio(iris_pts, corner_left, corner_right):
    iris_center = np.mean(iris_pts, axis=0)
    span = corner_right[0] - corner_left[0]
    if span == 0:
        return 0.5
    return (iris_center[0] - corner_left[0]) / span


def _head_pose(landmarks_px, image_size):
    h, w = image_size
    image_points = np.array([landmarks_px[i] for i in POSE_LANDMARK_IDS], dtype=np.float64)
    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    pose_mat = cv2.hconcat((rotation_mat, np.zeros((3, 1))))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = euler_angles.flatten()

    # decomposeProjectionMatrix wraps a frontal face to near +-180 on pitch;
    # fold it back so straight-ahead reads close to 0
    if pitch > 0:
        pitch = 180 - pitch
    elif pitch < 0:
        pitch = -180 - pitch

    return float(pitch), float(yaw), float(roll)


class FaceFeatureExtractor:
    def __init__(self, min_detection_confidence=0.3):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def extract(self, image_bgr):
        h, w = image_bgr.shape[:2]
        result = self.face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
        if not result.multi_face_landmarks:
            return None

        landmarks = result.multi_face_landmarks[0].landmark
        px = [(lm.x * w, lm.y * h) for lm in landmarks]

        left_ear = _eye_aspect_ratio([px[i] for i in LEFT_EYE])
        right_ear = _eye_aspect_ratio([px[i] for i in RIGHT_EYE])

        left_gaze = _gaze_ratio(
            [px[i] for i in LEFT_IRIS], px[LEFT_EYE_CORNERS[0]], px[LEFT_EYE_CORNERS[1]]
        )
        right_gaze = _gaze_ratio(
            [px[i] for i in RIGHT_IRIS], px[RIGHT_EYE_CORNERS[0]], px[RIGHT_EYE_CORNERS[1]]
        )

        pose = _head_pose(px, (h, w))
        if pose is None:
            pitch, yaw, roll = np.nan, np.nan, np.nan
        else:
            pitch, yaw, roll = pose

        return {
            "ear_left": left_ear,
            "ear_right": right_ear,
            "ear_mean": (left_ear + right_ear) / 2.0,
            "gaze_left_x": left_gaze,
            "gaze_right_x": right_gaze,
            "gaze_x_mean": (left_gaze + right_gaze) / 2.0,
            "head_pitch": pitch,
            "head_yaw": yaw,
            "head_roll": roll,
        }

    def close(self):
        self.face_mesh.close()
