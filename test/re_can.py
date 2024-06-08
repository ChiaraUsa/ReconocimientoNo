import cv2
import numpy as np
import mediapipe as mp
from collections import deque
from joblib import load
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Giving different arrays to handle colour points
points = [deque(maxlen=1024)]
index = 0

# The kernel to be used for dilation purpose
kernel = np.ones((5, 5), np.uint8)

# Setting color to black
color = (0, 0, 0)

# Load the model and scaler
model = load('../models/digit_recognizer')
scaler = load('../models/scaler.joblib')

# Function to predict digit
def prediction(image, model, scaler):
    img = cv2.resize(image, (28, 28))
    img = img.flatten().reshape(1, -1)
    img = scaler.transform(img)
    predict = model.predict(img)
    return predict[0]

# Here is code for Canvas setup
paintWindow = np.zeros((471, 636, 3)) + 255
paintWindow = cv2.rectangle(paintWindow, (40, 1), (140, 65), (0, 0, 0), 2)
cv2.putText(paintWindow, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
cv2.namedWindow('Paint', cv2.WINDOW_AUTOSIZE)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)
ret = True
while ret:
    # Read each frame from the webcam
    ret, frame = cap.read()

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    frame = cv2.rectangle(frame, (40, 1), (140, 65), (0, 0, 0), 2)
    cv2.putText(frame, "CLEAR", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)

    # Get hand landmark prediction
    result = hands.process(framergb)

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * 640)
                lmy = int(lm.y * 480)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)
        fore_finger = (landmarks[8][0], landmarks[8][1])
        center = fore_finger
        thumb = (landmarks[4][0], landmarks[4][1])
        cv2.circle(frame, center, 3, (0, 255, 0), -1)

        if (thumb[1] - center[1] < 30):
            points.append(deque(maxlen=512))
            index += 1
        elif center[1] <= 65:
            if 40 <= center[0] <= 140:  # Clear Button
                points = [deque(maxlen=512)]
                index = 0
                paintWindow[67:, :, :] = 255
        else:
            points[index].appendleft(center)
    else:
        points.append(deque(maxlen=512))
        index += 1

    # Draw lines on the canvas and frame
    for j in range(len(points)):
        for k in range(1, len(points[j])):
            if points[j][k - 1] is None or points[j][k] is None:
                continue
            cv2.line(frame, points[j][k - 1], points[j][k], color, 2)
            cv2.line(paintWindow, points[j][k - 1], points[j][k], color, 2)

    # Predict digit from the drawn image
    if index > 0 and len(points[index - 1]) > 1:
        bbox_size = (200, 200)
        bbox = [(int(x // 2 - bbox_size[0] // 2), int(y // 2 - bbox_size[1] // 2)),
                (int(x // 2 + bbox_size[0] // 2), int(y // 2 + bbox_size[1] // 2))]

        img_cropped = paintWindow[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]]
        img_cropped = np.array(img_cropped, dtype=np.uint8)  # Ensure the image is in uint8 format
        img_gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img_gray, 128, 255, cv2.THRESH_BINARY_INV)

        digit = prediction(thresh, model, scaler)

        # Add digit prediction to frame
        cv2.putText(frame, f'Digit: {digit}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Output", frame)
    cv2.imshow("Paint", paintWindow)

    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()
