import cv2
import time
from deepface import DeepFace

# Initialize webcam
cam = cv2.VideoCapture(0)  # 0 for default laptop webcam

# Time tracker for FPS
prev_frame_time = 0

while True:
    valid_frame, frame = cam.read()
    if not valid_frame:
        break

    # Resize frame for better speed
    resized_frame = cv2.resize(frame, (640, 480))

    try:
        # Analyze face
        results = DeepFace.analyze(
            resized_frame,
            actions=["age", "gender", "emotion"],
            enforce_detection=False,
            detector_backend='opencv'  # Fast and lightweight
        )

        # Extract predictions
        age = results[0]['age']
        gender = results[0]['dominant_gender']
        emotion = results[0]['dominant_emotion']
        region = results[0]['region']

        # Draw face box
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Add analysis text
        info = f"Age: {age} | Gender: {gender} | Mood: {emotion}"
        cv2.putText(frame, info, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    except Exception as e:
        cv2.putText(frame, "No face detected. Adjust the camera.",
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # FPS Calculation
    new_frame_time = time.time()
    fps = int(1 / (new_frame_time - prev_frame_time + 1e-5))
    prev_frame_time = new_frame_time

    cv2.putText(frame, f"FPS: {fps}", (20, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # Show video
    cv2.imshow("Face Analysis - Press 'q' to exit", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cam.release()
cv2.destroyAllWindows()
