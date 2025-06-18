import cv2
import mediapipe as mp
import turtle
import pyautogui
import random
import time
import numpy as np

# Disable pyautogui failsafe (prevents crashing if mouse moves to a screen corner)
pyautogui.FAILSAFE = False

# Get screen size
screen_w, screen_h = pyautogui.size()

# Setup Turtle screen
sc = turtle.Screen()
sc.setup(width=1.0, height=1.0)
sc.bgcolor("white")

# Display instruction
pen = turtle.Turtle()
pen.hideturtle()
pen.penup()
pen.goto(0, 0)
pen.write("Calibration Window - Center your face to the camera and stare at the dot", align="center", font=("Arial", 16, "normal"))
time.sleep(5.0)
sc.clear()

# Draw red dot
dot = turtle.Turtle()
dot.penup()
dot.shape("circle")
dot.color("red")
dot.shapesize(2)

# Define calibration points using user's screen size and adding a margin
calibration_points = [
    (-screen_w/2+20, screen_h/2-25), (0, screen_h/2-25), (screen_w/2-20, screen_h/2-25),
    (-screen_w/2+20, 0),   (0, 0),   (screen_w/2-20, 0),
    (-screen_w/2+20, -screen_h/2+25), (0, -screen_h/2+25), (screen_w/2-20, -screen_h/2+25)
]


calibration_data = []

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
capture = cv2.VideoCapture(0)

# Calibration process
for point in calibration_points:
    dot.goto(point)
    dot.stamp()
    print(f"Look at: {point}")
    time.sleep(2)

    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        continue
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        iris = face_landmarks.landmark[468]
        iris_x = int(iris.x * w)
        iris_y = int(iris.y * h)

        calibration_data.append({
            "screen": point,
            "iris": (iris_x, iris_y)
        })
        print(f"Captured iris at: {(iris_x, iris_y)} for screen point {point}")

# Function to predict screen position from iris
def predict_screen_position(new_iris, calibration_data):
    iris_coords = np.array([data["iris"] for data in calibration_data])
    screen_coords = np.array([data["screen"] for data in calibration_data])

    # Add bias column
    X = np.hstack((iris_coords, np.ones((len(iris_coords), 1))))
    A, _, _, _ = np.linalg.lstsq(X, screen_coords, rcond=None)

    new_input = np.array([new_iris[0], new_iris[1], 1])
    predicted = new_input @ A
    return predicted

# Hide turtle screen
turtle.bye()

# Begin real-time eye tracking
print("Starting real-time eye tracking. Press 'q' in the webcam window to quit.")
current_x, current_y = pyautogui.position()

# dwell-to-click setup
dwell_start_time = None
dwell_threshold = 1.5  # seconds
click_position = None
tolerance = 50  # pixels

previous_iris = None

while True:
    ret, frame = capture.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0]
        h, w, _ = frame.shape
        iris = face_landmarks.landmark[468]
        iris_x = int(iris.x * w)
        iris_y = int(iris.y * h)

        predicted = predict_screen_position((iris_x, iris_y), calibration_data)

        # Convert to screen coordinates (top-left = (0,0))
        screen_x = predicted[0] + screen_w / 2
        screen_y = -predicted[1] + screen_h / 2

        # Adjust sensitivity based on iris movement
        if previous_iris is not None:
            movement = np.linalg.norm(np.array([iris_x, iris_y]) - np.array(previous_iris))
            if movement < 5:
                alpha = 0.02  # lower alpha = slower, more precise
            else:
                alpha = 0.08  # default speed
        else:
            alpha = 0.08

        previous_iris = (iris_x, iris_y)

        # Smooth interpolation toward predicted point
        current_x = (1 - alpha) * current_x + alpha * screen_x
        current_y = (1 - alpha) * current_y + alpha * screen_y

        pyautogui.moveTo(current_x, current_y, _pause=False)

        # Dwell-to-click logic
        if click_position is None or np.linalg.norm(np.array([current_x, current_y]) - np.array(click_position)) > tolerance:
            click_position = (current_x, current_y)
            dwell_start_time = time.time()
        else:
            if time.time() - dwell_start_time > dwell_threshold:
                pyautogui.click()
                dwell_start_time = time.time() + 1  # avoid repeat clicking

    # Show webcam feed
    cv2.imshow("Eye Tracking - Press 'q' to Quit", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
