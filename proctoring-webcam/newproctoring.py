import cv2
import mediapipe as mp
import time
import math

# --- Configuration ---
# Head Deviation: Max horizontal pixel distance the nose can be from the center.
HEAD_DEVIATION_THRESHOLD_PX = 100 

# Gaze Deviation: Normalized threshold (0.0 to 1.0) for iris shift relative to eye width.
GAZE_DEVIATION_THRESHOLD = 0.15 

# FIX: Lowering the threshold to 0.75.
# This means the anomaly is triggered if the hand is in the top 75% of the screen (Y < 0.75 normalized).
HAND_PROXIMITY_Y_THRESHOLD = 0.75 

ANOMALY_TRIGGER_TIME = 10.0  # Time in seconds (e.g., 10.0 seconds)

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Define drawing_spec
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize the Face Mesh model with refined landmarks for iris tracking
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,  # Crucial for iris tracking
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Initialize the Hands model
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Landmark Indices
NOSE_TIP = 1
# Right Eye landmarks (iris center)
R_IRIS_CENTER = 473 
R_EYE_LEFT = 33
R_EYE_RIGHT = 133

# Left Eye landmarks (iris center)
L_IRIS_CENTER = 468
L_EYE_LEFT = 362
L_EYE_RIGHT = 263

# Hand Landmark: Index Finger Tip (Landmark 8) for better phone/note proximity detection
INDEX_FINGER_TIP = mp_hands.HandLandmark.INDEX_FINGER_TIP.value


# Global variables for tracking time
start_time_deviation = None
is_anomalous = False
anomaly_triggered = False

print("\n--- Proctoring System V5 Initiated (Hand Threshold: 0.75) ---")
print(f"Monitoring Head Position, Gaze, and Hand Proximity. Trigger: {ANOMALY_TRIGGER_TIME}s")
print("Press 'q' to exit the video feed.\n")


def check_gaze(face_landmarks, w, h):
    """Calculates if the gaze is centered based on iris position."""
    
    # Check if iris landmarks are present 
    if R_IRIS_CENTER >= len(face_landmarks.landmark) or L_IRIS_CENTER >= len(face_landmarks.landmark):
        return False, 0.0 
        
    # 1. Right Eye Gaze Check
    r_iris = face_landmarks.landmark[R_IRIS_CENTER]
    r_eye_l = face_landmarks.landmark[R_EYE_LEFT]
    r_eye_r = face_landmarks.landmark[R_EYE_RIGHT]
    
    # Horizontal center of the right eye box (normalized)
    r_eye_center = (r_eye_l.x + r_eye_r.x) / 2
    
    # Normalized horizontal distance from iris to eye center
    r_gaze_dev = abs(r_iris.x - r_eye_center)
    
    # 2. Left Eye Gaze Check
    l_iris = face_landmarks.landmark[L_IRIS_CENTER]
    l_eye_l = face_landmarks.landmark[L_EYE_LEFT]
    l_eye_r = face_landmarks.landmark[L_EYE_RIGHT]
    
    # Horizontal center of the left eye box (normalized)
    l_eye_center = (l_eye_l.x + l_eye_r.x) / 2
    
    # Normalized horizontal distance from iris to eye center
    l_gaze_dev = abs(l_iris.x - l_eye_center)

    # Use the maximum deviation as the primary metric
    max_gaze_dev = max(r_gaze_dev, l_gaze_dev)
    
    is_gaze_deviating = max_gaze_dev > GAZE_DEVIATION_THRESHOLD
    
    return is_gaze_deviating, max_gaze_dev

def check_hands(hand_landmarks, h):
    """Checks if any hand is in the upper 75% of the screen (suspicious zone) 
       by checking the index finger tip position."""
    
    if not hand_landmarks:
        return False
    
    is_hand_suspicious = False
    for hand in hand_landmarks:
        # Check the index finger tip position (Landmark 8)
        index_finger_tip = hand.landmark[INDEX_FINGER_TIP] 
        
        # Check if the index finger tip is above the proximity threshold (Y < 0.75 normalized)
        if index_finger_tip.y < HAND_PROXIMITY_Y_THRESHOLD:
            is_hand_suspicious = True
            break
            
    return is_hand_suspicious


while cap.isOpened():
    success, image = cap.read()
    if not success:
        continue

    # Flip the image horizontally for a selfie-view display and convert BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with both models
    face_results = face_mesh.process(image_rgb)
    hand_results = hands.process(image_rgb)

    h, w, _ = image.shape
    center_x = w // 2
    current_time = time.time()
    
    
    # --- ANOMALY FLAGS (Default to safe) ---
    is_head_deviating = False
    is_gaze_deviating = False
    is_hand_detected = False
    gaze_dev_score = 0.0
    
    # --- 1. HEAD AND GAZE CHECK ---
    if face_results.multi_face_landmarks:
        face_landmarks = face_results.multi_face_landmarks[0]
        
        # A. Head Deviation Check (Existing Logic)
        nose_tip = face_landmarks.landmark[NOSE_TIP]
        nose_x_px = int(nose_tip.x * w)
        horizontal_deviation = abs(nose_x_px - center_x)
        if horizontal_deviation > HEAD_DEVIATION_THRESHOLD_PX:
            is_head_deviating = True
            
        # B. Gaze Deviation Check (New Logic)
        is_gaze_deviating, gaze_dev_score = check_gaze(face_landmarks, w, h)
        
        # Draw all face landmarks
        mp_drawing.draw_landmarks(
            image,
            face_landmarks,
            mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec
        )

        # Draw nose marker based on head deviation
        nose_color = (0, 0, 255) if is_head_deviating else (0, 255, 0)
        cv2.circle(image, (nose_x_px, int(nose_tip.y * h)), 5, nose_color, -1)
        
    else:
        # No face detected is also an anomaly
        is_head_deviating = True 
        
    
    # --- 2. HAND/OBJECT CHECK (Improved Threshold) ---
    if hand_results.multi_hand_landmarks:
        is_hand_detected = check_hands(hand_results.multi_hand_landmarks, h)
        
        # Draw hand landmarks
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(255,255,0), thickness=2, circle_radius=2), # Cyan dots
                mp_drawing.DrawingSpec(color=(0,128,255), thickness=2) # Orange lines
            )
        
    # --- 3. COMBINED ANOMALY LOGIC ---
    current_anomaly_state = is_head_deviating or is_gaze_deviating or is_hand_detected
    
    if current_anomaly_state:
        if not is_anomalous:
            # Transition to anomalous state
            start_time_deviation = current_time
            is_anomalous = True
            anomaly_triggered = False
            
        elapsed_time = current_time - start_time_deviation
        
        # Determine the primary cause for display
        cause = ""
        if not face_results.multi_face_landmarks:
             cause = "No Face"
        elif is_head_deviating:
             cause = "Head Deviation"
        elif is_gaze_deviating:
             cause = f"Gaze Shift ({gaze_dev_score:.2f})"
        elif is_hand_detected:
             cause = "Hand Proximity (Phone/Note)"
        
        status_text = f"ANOMALY: {cause} | Time: {elapsed_time:.1f}s"
        status_color = (0, 0, 255) # Red (BGR format)
        
        if elapsed_time >= ANOMALY_TRIGGER_TIME:
            if not anomaly_triggered:
                print(f"\n[CRITICAL ANOMALY] Total time exceeded {ANOMALY_TRIGGER_TIME}s. Cause: {cause}")
                anomaly_triggered = True
            status_text = "CRITICAL: ANOMALY LOGGED!"
            
    else:
        # Safe State
        is_anomalous = False
        start_time_deviation = None
        anomaly_triggered = False
        
        status_text = "STATUS: Monitoring (Safe)"
        status_color = (0, 255, 0) # Green (BGR format)

    # --- Drawing Status and Guides ---
    cv2.putText(image, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    # Draw central guide lines
    cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 1) 
    # Draw deviation boundary (red lines)
    cv2.line(image, (center_x - HEAD_DEVIATION_THRESHOLD_PX, 0), (center_x - HEAD_DEVIATION_THRESHOLD_PX, h), (0, 0, 255), 1)
    cv2.line(image, (center_x + HEAD_DEVIATION_THRESHOLD_PX, 0), (center_x + HEAD_DEVIATION_THRESHOLD_PX, h), (0, 0, 255), 1)
    # Draw hand proximity boundary (yellow line)
    # The yellow line should now appear lower on the screen (at Y=0.75)
    cv2.line(image, (0, int(h * HAND_PROXIMITY_Y_THRESHOLD)), (w, int(h * HAND_PROXIMITY_Y_THRESHOLD)), (0, 255, 255), 1)


    # Display the final image
    cv2.imshow('Real-Time Proctoring Core V5', image)

    # Press 'q' to exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
face_mesh.close()
hands.close()
cap.release()
cv2.destroyAllWindows()
