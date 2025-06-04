import cv2
import mediapipe as mp
import turtle
import random
import time

# Setup Turtle Screen
sc = turtle.Screen()
sc.setup(width=1.0, height=1.0)
sc.bgcolor("white")

# Text Display for 5 seconds
pen = turtle.Turtle()
pen.hideturtle()
pen.penup()
pen.goto(0, 0)
pen.write("Calibration Window - Center your face to the camera and stare at the dot", align="center", font=("Arial", 16, "normal"))
time.sleep(5.0)
sc.clear()

# draw red dot
dot = turtle.Turtle()
dot.penup()
dot.shape("circle")
dot.color("red")
dot.shapesize(2)

# Define calibration points (in screen coordinates)
calibration_points = [
    (-700, 400), (0, 400), (700, 400),
    (-700, 0),   (0, 0),   (700, 0),
    (-700, -400), (0, -400), (700, -400)
]

# Store calibration data
calibration_data = []

# Setup MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)  # enables iris tracking
mp_drawing = mp.solutions.drawing_utils

# Setup Webcam
capture = cv2.VideoCapture(1)
if not capture.isOpened():
    print("Error: Cannot open webcam")
    exit()

# Main loop for processing video frames
for point in calibration_points:
    dot.goto(point)
    dot.stamp()
    print(f"Look at: {point}")
    time.sleep(2)

    ret, frame = capture.read()
    if not ret:
        print("Failed to grab frame")
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    # If face landmarks are found
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            h, w, _ = frame.shape
            # Right iris center (landmark 468)
            iris = face_landmarks.landmark[468]
            iris_x = int(iris.x * w)
            iris_y = int(iris.y * h)

            calibration_data.append({
                "screen": point,
                "iris": (iris_x, iris_y)
            })
            print(f"Captured iris at: {(iris_x, iris_y)} for screen point {point}")
            
    print("Calibration complete. Data:")
    for data in calibration_data:
        print(data)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
turtle.bye()
