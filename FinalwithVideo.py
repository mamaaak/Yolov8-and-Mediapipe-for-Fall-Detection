# Import necessary libraries
from ultralytics import YOLO
import cv2
import cvzone
import math
import mediapipe as mp
import numpy as np
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import winsound  # Import winsound for playing alert sound

# Function to send email
def send_email():
    print("Attempting to send email...")  # Add a print statement to indicate the function is called
    # Email configuration
    sender_email = "qmaarapelo@tip.edu.ph"  # Sender's email address
    receiver_email = ""  # Receiver's email address
    password = ""  # Sender's email password

    # Email content
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Fall Detected!"

    body = "A fall has been detected. Please check the fall detection system for details."
    message.attach(MIMEText(body, "plain"))

    try:
        # Send email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:  # SMTP server configuration
            server.login(sender_email, "clvjibdzczuzxjag")
            server.sendmail(sender_email, receiver_email, message.as_string())
        print("Email sent successfully!")  # Add a print statement to indicate successful email sending
    except Exception as e:
        print("Error sending email:", e)  # Add a print statement to print out the error message

# Initialize YOLO model
model = YOLO("best (1).pt")

classNames = ['person']

# Initialize Mediapipe pose estimation
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Additional parameters for refined fall detection
fall_velocity_threshold = 0.5  # Threshold for the speed of change in landmark positions
prev_landmark_positions = None  # To store previous frame's landmark positions
frame_count = 0  # Counter for frames since the start of a potential fall

# Initialize variables and parameters from the first code snippet
counter = 0
fall_detected = False
prev_fall_status = False
font = cv2.FONT_HERSHEY_SIMPLEX
body_angle = 'front'
sideway_slight = 0
sideway_whole = 0
front = 0
fall = 0

# Main loop to process each frame from the video file
video_path = "50ways.mp4"  # Replace "path_to_your_video_file.mp4" with the actual path to your video file
cap = cv2.VideoCapture(video_path)

# Check if video file opened successfully
if not cap.isOpened():
    print("Error: Unable to open video file.")
    exit()

# Define a fixed position for the overlay text
text_position = (10, 50)  # Adjust the coordinates as needed

# Main loop to process each frame from the webcam
with mp_pose.Pose(min_detection_confidence=0.6, min_tracking_confidence=0.6) as pose:
    while cap.isOpened():
        ret, img = cap.read()
        if not ret:
            print("Failed to read frame from the video file")
            break

        # Reset fall detected flag for each frame
        fall_detected_flag = False

        # Perform object detection using YOLO
        results = model(img, stream=True)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if classNames[cls] == 'person':
                    conf = box.conf[0]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f'{classNames[cls]} {conf:.2f}', (max(0, x1), max(35, y1)),
                                       scale=0.9, thickness=2)

                    # Perform pose estimation using Mediapipe on each detected person
                    crop_img = img[y1:y2, x1:x2]
                    crop_img.flags.writeable = False
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    results = pose.process(crop_img)

                    # Inside the main loop, after pose landmarks are detected
                    if results.pose_landmarks:
                        lst = []
                        for landmark in results.pose_landmarks.landmark:
                            lst.append((landmark.x, landmark.y, landmark.z, landmark.visibility))
                        # Calculate velocity of change in landmark positions if previous positions are available
                        if prev_landmark_positions:
                            velocities = [np.linalg.norm(np.array(curr) - np.array(prev)) for curr, prev in
                                          zip(lst, prev_landmark_positions)]
                            max_velocity = max(velocities)
                        else:
                            max_velocity = 0

                        # Calculate the angle of the body
                        shoulder_center = np.array([(lst[11][0] + lst[12][0]) / 2, (lst[11][1] + lst[12][1]) / 2])
                        hip_center = np.array([(lst[23][0] + lst[24][0]) / 2, (lst[23][1] + lst[24][1]) / 2])
                        body_angle = np.arctan2(hip_center[1] - shoulder_center[1],
                                                 hip_center[0] - shoulder_center[0])

                        # Update previous landmark positions
                        prev_landmark_positions = lst

                        s_h_high = abs((lst[23][1] + lst[24][1] - lst[11][1] - lst[12][1]) / 2)
                        s_h_long = np.sqrt(
                            ((lst[23][1] + lst[24][1] - lst[11][1] - lst[12][1]) / 2) ** 2 +
                            ((lst[23][0] + lst[24][0] - lst[11][0] - lst[12][0]) / 2) ** 2)
                        h_f_high = ((lst[28][1] + lst[27][1] - lst[24][1] - lst[23][1]) / 2)
                        h_f_long = np.sqrt(
                            ((lst[28][1] + lst[27][1] - lst[24][1] - lst[23][1]) / 2) ** 2 +
                            ((lst[28][0] + lst[27][0] - lst[24][0] - lst[23][0]) / 2) ** 2)

                        # Define fall detection algorithm parameters
                        para_s_h_1 = 1.15
                        para_s_h_2 = 0.85
                        para_h_f = 0.6
                        para_fall_time = 5

                        # Perform fall detection
                        if abs(body_angle) > np.radians(75):
                            if s_h_high < s_h_long * para_s_h_1 and s_h_high > s_h_long * para_s_h_2:
                                fall_detected = False
                            elif h_f_high < para_h_f * h_f_long:
                                counter += 1
                                if counter >= para_fall_time:
                                    fall_detected = True
                                    counter = 0
                                else:
                                    fall_detected = False
                            else:
                                counter = 0
                                fall_detected = False

                        # Update fall detected flag if a fall is detected in any person
                        if fall_detected:
                            fall_detected_flag = True

                    # Update frame with overlays
                    crop_img.flags.writeable = True
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR)
                    mp_drawing.draw_landmarks(crop_img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2,
                                                                      circle_radius=2),
                                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2,
                                                                      circle_radius=2))

                    # Place the cropped image back into the original image
                    img[y1:y2, x1:x2] = crop_img

        # If a fall is detected in any person, display "Fall Detected" text on the frame
        if fall_detected_flag:
            cv2.putText(img, 'Fall Detected!', text_position, font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            send_email()  # Call the send_email function if fall detected
            winsound.PlaySound("alert.wav", winsound.SND_FILENAME)  # Play the alert sound
        else:
            # Display "No Fall Detected" text only if no fall is detected in any person
            cv2.putText(img, 'No Fall Detected', text_position, font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # Show the frame
        cv2.imshow('Fall Detection Feed', img)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
