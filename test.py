import cv2
import mediapipe as mp
import turtle
import random

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # enables iris tracking
mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

sc = turtle.Screen()
sc.setup(width=1.0, height=1.0)
sc.bgcolor("white")

while True:
    ret, frame = capture.read()
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw all landmarks
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)

            # Get right and left iris center (landmarks 468 and 473)
            h, w, _ = frame.shape
            for idx in [468, 473]:  # right and left iris center
                x = int(face_landmarks.landmark[idx].x * w)
                y = int(face_landmarks.landmark[idx].y * h)
                cv2.circle(frame, (x, y), 5, (0, 255, 255), -1)  # Draw iris center

    cv2.imshow("MediaPipe Iris Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()

