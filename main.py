import cv2
import time
from deepface import DeepFace

cam = cv2.VideoCapture(0)  # 0 for default laptop webcam

# For calculating FPS
prev_frame_rate = 0
new_frame_rate = 0

# while true kyuki - to continue till we end manually
while True: 
    valid_frame, frame = cam.read()
    if not valid_frame:
        break

    resized_frame = cv2.resize(frame, (640, 480))
    new_frame_time = time.time()

    try:
        # Analysis of face will be here
        results = DeepFace.analyze(
            resized_frame,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
            detector_backend='opencv'
        )

        # Extract predictions
        age = results[0]['age']
        gender = results[0]['dominant_gender']
        emotion = results[0]['dominant_emotion']
        region = results[0]['region']

        # Face box dimensions here
        x, y, w, h = region['x'], region['y'], region['w'], region['h']

        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Prediction text
        info = f"Age: {age} | Gender: {gender} | Mood: {emotion}"
        cv2.putText(frame, info, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, "No face detected. Adjust the camera.",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS calculation
    fps = int(1 / (new_frame_rate - prev_frame_rate + 1e-5))
    prev_frame_rate = new_frame_rate
    cv2.putText(frame, f"FPS: {fps}", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Shows video
    cv2.imshow("Face Analysis - Press 'q' to exit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

