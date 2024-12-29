import cv2
import numpy as np
import tensorflow as tf
import dlib
import pygame
import time
from threading import Thread
from scipy.spatial import distance as dist

# Initialize pygame for audio
pygame.mixer.init()
alarm_sound = pygame.mixer.Sound('../Sound/alarm.wav')

# Load the models
eye_model = tf.keras.models.load_model('../Models/simple_CNN_eye_modelV6_3.keras')
mouth_model = tf.keras.models.load_model('../Models/simple_CNN_mouth_modelV6_3.keras')

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('../Models/shape_predictor_68_face_landmarks.dat')

# State dictionaries
state_dict1 = {0: "Eyes Closed", 1: "Eyes Open"}
state_dict2 = {0: "No Yawn", 1: "Yawn"}

def enhance_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l,a,b))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr

def play_alarm(volume=0.7, duration=None):
    alarm_sound.set_volume(volume)
    alarm_sound.play()
    if duration:
        time.sleep(duration)
        alarm_sound.stop()

def get_eyes_mouth_roi(frame, shape):
    # Convert shape to numpy array
    shape_np = np.zeros((68, 2), dtype=int)
    for i in range(0, 68):
        shape_np[i] = (shape.part(i).x, shape.part(i).y)
    
    # Extract coordinates
    left_eye = shape_np[36:42]
    right_eye = shape_np[42:48]
    mouth = shape_np[48:68]
    
    # Get convex hulls
    left_eye_hull = cv2.convexHull(left_eye)
    right_eye_hull = cv2.convexHull(right_eye)
    mouth_hull = cv2.convexHull(mouth)
    
    # Get bounding rectangles
    padding = 10
    
    # Left eye ROI
    left_eye_rect = cv2.boundingRect(left_eye_hull)
    left_eye_roi = frame[
        max(left_eye_rect[1]-padding, 0):min(left_eye_rect[1]+left_eye_rect[3]+padding, frame.shape[0]),
        max(left_eye_rect[0]-padding, 0):min(left_eye_rect[0]+left_eye_rect[2]+padding, frame.shape[1])
    ]
    
    # Right eye ROI
    right_eye_rect = cv2.boundingRect(right_eye_hull)
    right_eye_roi = frame[
        max(right_eye_rect[1]-padding, 0):min(right_eye_rect[1]+right_eye_rect[3]+padding, frame.shape[0]),
        max(right_eye_rect[0]-padding, 0):min(right_eye_rect[0]+right_eye_rect[2]+padding, frame.shape[1])
    ]
    
    # Mouth ROI
    mouth_rect = cv2.boundingRect(mouth_hull)
    mouth_roi = frame[
        max(mouth_rect[1]-padding, 0):min(mouth_rect[1]+mouth_rect[3]+padding, frame.shape[0]),
        max(mouth_rect[0]-padding, 0):min(mouth_rect[0]+mouth_rect[2]+padding, frame.shape[1])
    ]
    
    return left_eye_roi, right_eye_roi, mouth_roi, left_eye_hull, right_eye_hull, mouth_hull

def calculate_EAR(eye_points):
    eye_points = np.array(eye_points)
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_eyes(frame, eyes_roi):
    if eyes_roi.size == 0:
        return "Unknown"
    gray_eye = cv2.cvtColor(eyes_roi, cv2.COLOR_BGR2GRAY)
    resized_eye = cv2.resize(gray_eye, (128, 128))
    normalized_eye = resized_eye / 255.0
    eye_input = np.expand_dims(normalized_eye, axis=[0, -1])
    prediction = eye_model.predict(eye_input, verbose=0)
    state = np.argmax(prediction)
    confidence = prediction[0][state] * 100
    return state_dict1[state], confidence

def process_mouth(frame, mouth_roi):
    if mouth_roi.size == 0:
        return "Unknown"
    gray_mouth = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
    resized_mouth = cv2.resize(gray_mouth, (256, 256))
    normalized_mouth = resized_mouth / 255.0
    mouth_input = np.expand_dims(normalized_mouth, axis=[0, -1])
    prediction = mouth_model.predict(mouth_input, verbose=0)
    state = np.argmax(prediction)
    confidence = prediction[0][state] * 100
    return state_dict2[state], confidence

def draw_status(frame, left_eye_state, right_eye_state, mouth_state, avg_ear, 
                left_conf, right_conf, mouth_conf, status="Normal"):
    # Background for text
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    
    # Display EAR
    cv2.putText(frame, f"EAR: {avg_ear:.2f}", (15, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display eye states
    cv2.putText(frame, f"Left Eye: {left_eye_state} ({left_conf:.1f}%)", (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(frame, f"Right Eye: {right_eye_state} ({right_conf:.1f}%)", (15, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display mouth state
    cv2.putText(frame, f"Mouth: {mouth_state} ({mouth_conf:.1f}%)", (15, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Display status
    color = (0, 255, 0) if status == "Normal" else (0, 0, 255)
    cv2.putText(frame, f"Status: {status}", (15, 135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

def main():
    cap = cv2.VideoCapture(0)
    
    # State variables
    eyes_closed_start_time = None
    face_not_visible_start_time = None
    alarm_active = False
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Enhance contrast
        enhanced_frame = enhance_contrast(frame)
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = detector(gray)
        
        current_status = "Normal"
        
        if faces:
            face_not_visible_start_time = None
            if alarm_active and pygame.mixer.get_busy():
                alarm_sound.stop()
                alarm_active = False
            
            for face in faces:
                # Get facial landmarks
                shape = predictor(gray, face)
                
                # Convert shape to numpy array for EAR calculation
                shape_np = np.zeros((68, 2), dtype=int)
                for i in range(0, 68):
                    shape_np[i] = (shape.part(i).x, shape.part(i).y)
                
                # Get ROIs and hulls
                left_eye_roi, right_eye_roi, mouth_roi, left_eye_hull, right_eye_hull, mouth_hull = get_eyes_mouth_roi(frame, shape)
                
                # Calculate EAR
                left_ear = calculate_EAR(shape_np[36:42])
                right_ear = calculate_EAR(shape_np[42:48])
                avg_ear = (left_ear + right_ear) / 2.0
                
                # Process eyes and mouth
                left_eye_state, left_conf = process_eyes(frame, left_eye_roi)
                right_eye_state, right_conf = process_eyes(frame, right_eye_roi)
                mouth_state, mouth_conf = process_mouth(frame, mouth_roi)
                
                # Case 1: Yawning with closed eyes
                if mouth_state == "Yawn" and \
                   left_eye_state == "Eyes Closed" and \
                   right_eye_state == "Eyes Closed":
                    Thread(target=play_alarm, args=(0.7, 5)).start()
                    current_status = "Warning: Yawning with Closed Eyes!"
                
                # Case 2: Both eyes closed for 5 seconds
                EAR_THRESHOLD = 0.2
                if avg_ear < EAR_THRESHOLD and \
                   left_eye_state == "Eyes Closed" and \
                   right_eye_state == "Eyes Closed":
                    if eyes_closed_start_time is None:
                        eyes_closed_start_time = time.time()
                    elif time.time() - eyes_closed_start_time >= 3:
                        Thread(target=play_alarm, args=(1.0,)).start()
                        current_status = "Warning: Eyes Closed for Too Long!"
                else:
                    eyes_closed_start_time = None
                    if pygame.mixer.get_busy():
                        alarm_sound.stop()
                
                # Draw the eye and mouth regions
                cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
                cv2.drawContours(frame, [mouth_hull], -1, (0, 255, 0), 1)
                
                # Draw status
                draw_status(frame, left_eye_state, right_eye_state, mouth_state, 
                          avg_ear, left_conf, right_conf, mouth_conf, current_status)
        
        else:
            # Case 3: Face not visible
            if face_not_visible_start_time is None:
                face_not_visible_start_time = time.time()
            elif time.time() - face_not_visible_start_time >= 1:
                if not alarm_active:
                    Thread(target=play_alarm, args=(0.7,)).start()
                    alarm_active = True
                current_status = "Warning: Face Not Detected!"
            
            # Draw status when no face is detected
            draw_status(frame, "Unknown", "Unknown", "Unknown", 
                       0.0, 0.0, 0.0, 0.0, current_status)
        
        # Display the frame
        cv2.imshow('Drowsy Driver Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()