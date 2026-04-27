from pathlib import Path
import cv2
import time
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from playsound3 import playsound
import numpy as np

delay = 1

def get_angle(a,b,c):
    cb = np.atan2(c[1] - b[1], c[0] - b[0])
    ab = np.atan2(a[1] - b[1], a[0] - b[0])
    angle = np.rad2deg(cb - ab)
    angle = angle + 360 if angle < 0 else angle
    return 360 - angle if angle > 180 else angle

# def detect_hands_up(annotated, keypoints):
#     nose_seen = keypoints[0][0] > 0 and keypoints[0][1] > 0
#     eyes_seen = (keypoints[1][0] > 0 and keypoints[1][1] > 0) and (keypoints[2][0] > 0 and keypoints[2][1] > 0)
#     left_shoulder = keypoints[5]
#     right_shoulder = keypoints[6]
#     left_elbow = keypoints[7]
#     right_elbow = keypoints[8]
#     left_wrist = keypoints[9]
#     right_wrist = keypoints[10]
#     if nose_seen and eyes_seen:
#         if (left_shoulder[1] > left_elbow[1] > left_wrist[1]) and  \
#         (right_shoulder[1] > right_elbow[1] > right_wrist[1]):
#             left_angle = get_angle(left_shoulder, left_elbow, left_wrist)
#             cv2.putText(annotated, f"Hands Up({left_angle:.1f})", (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),1)
#             return True
#     cv2.putText(annotated, "", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0),1)
#     return None

def detect_push_up(annotated, keypoints, pushups, pushup_state, last_pushup_time):
    left_shoulder = keypoints[5]
    right_shoulder = keypoints[6]
    left_elbow = keypoints[7]
    right_elbow = keypoints[8]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]

    left_angle = get_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = get_angle(right_shoulder, right_elbow, right_wrist)

    avg_angle = (left_angle + right_angle) / 2

    current_time = time.time()

    if avg_angle < 100:
        pushup_state = "down"

    if avg_angle > 145 and pushup_state == "down":
        if current_time - last_pushup_time > delay:
            pushups += 1
            last_pushup_time = current_time
            pushup_state = "up"

    cv2.putText(
        annotated,
        f"Push-ups: {pushups}",
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.5,
        (0, 255, 0),
        2
    )

    return pushups, pushup_state, last_pushup_time

model = YOLO("yolo26n-pose.pt")
camera = cv2.VideoCapture("push_ups.mp4")

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = camera.get(cv2.CAP_PROP_FPS)

writer = cv2.VideoWriter("result_push_ups.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))


cv2.namedWindow("Pose", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Pose", 640, 480)

ps = None
pushups = 0
pushup_state = None
last_pushup_time = 0


while camera.isOpened():
    ret, frame = camera.read()
    if not ret:
        break
    # cv2.imshow("Camera", frame)
    key = cv2.waitKey(10) & 0xFF
    if key == ord("q"):
        break
    
    # t = time.perf_counter()
    results = model(frame)
    # print(f"Elapsed time {time.perf_counter() - t}, FPS {1 / (time.perf_counter() - t):.1f}")
    if not results:
        continue
    result = results[0]
    keypoints = result.keypoints.xy.tolist()
    if not keypoints:
        continue
    # print(keypoints)

    annotator = Annotator(frame)
    annotator.kpts(result.keypoints.data[0], result.orig_shape, 5, True)
    annotated = annotator.result()
    # if(detect_hands_up(annotated, keypoints[0])):
    #     if ps is None:
    #         ps = playsound("acolyteyes2.mp3", block = False)
    #     else:
    #         if not ps.is_alive():
    #             ps = None

    pushups, pushup_state, last_pushup_time = detect_push_up(annotated, keypoints[0], pushups, pushup_state, last_pushup_time)

    cv2.imshow("Pose", annotated)
    writer.write(annotated)

camera.release()
writer.release()
cv2.destroyAllWindows()