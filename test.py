import cv2
import numpy as np

# Start Webcam
cap = cv2.VideoCapture(0)

# Load face and eye detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while True:
    ret, frame = cap.read()  # Read frame from webcam
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)  # Detect faces

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)  # Draw face box
        roi_gray = gray[y:y + w, x:x + w]
        roi_color = frame[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)  # Detect eyes in face
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)  # Draw eye boxes

    cv2.imshow('frame', frame)  # Show result

    if cv2.waitKey(1) == ord('q'):  # Button to release camera
        break
cap.release()
cv2.destroyAllWindows()  #End program by closing screen capture window
