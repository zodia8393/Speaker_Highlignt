import numpy as np

def get_mouth_aspect_ratio(landmarks):
    if len(landmarks) < 68:
        return 0
    left = landmarks[61]
    right = landmarks[291]
    top = ((landmarks[37][0] + landmarks[267][0]) / 2, (landmarks[37][1] + landmarks[267][1]) / 2)
    bottom = ((landmarks[84][0] + landmarks[314][0]) / 2, (landmarks[84][1] + landmarks[314][1]) / 2)
    mar = (abs(top[1] - bottom[1])) / abs(left[0] - right[0])
    return mar

def get_movement_ratio(landmarks):
    if not landmarks:
        return 0
    movement_sum = 0
    num_points = len(landmarks)
    for i in range(num_points):
        for j in range(i + 1, num_points):
            movement_sum += np.linalg.norm(np.array(landmarks[i]) - np.array(landmarks[j]))
    movement_ratio = movement_sum / (num_points * (num_points - 1) / 2)
    return movement_ratio

def check_hand_over_mouth(face_landmarks, hand_landmarks):
    if not face_landmarks or not hand_landmarks:
        return False
    mouth_center = ((face_landmarks[13][0] + face_landmarks[14][0]) / 2, (face_landmarks[13][1] + face_landmarks[14][1]) / 2)
    for landmark in hand_landmarks:
        if abs(landmark[0] - mouth_center[0]) < 20 and abs(landmark[1] - mouth_center[1]) < 20:
            return True
    return False

def apply_highlight(frame, mask, color, alpha=0.5):
    overlay = frame.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
