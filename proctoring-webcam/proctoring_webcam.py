import cv2
import mediapipe as mp
import time
import math

# --- Configuration ---
# Threshold: The maximum allowed pixel distance (horizontal) the nose tip can be 
# from the center of the screen before being flagged as a deviation.
# This value (e.g., 100 pixels) will need tuning based on camera resolution/distance.
DEVIATION_THRESHOLD_PIXELS = 100 
ANOMALY_TRIGGER_TIME = 10.0  # Time in seconds (e.g., 10.0 seconds)

# --- MediaPipe Setup ---
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Initialize the Face Mesh model
# min_detection_confidence is set high for robustness
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75
)

# Initialize webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Global variables for tracking time
start_time_deviation = None
is_deviating = False
anomaly_triggered = False

print("\n--- Proctoring System Initiated ---")
print(f"Threshold: {ANOMALY_TRIGGER_TIME} seconds of deviation.")
print("Look away or turn your head to test the detection logic.\n")

# Landmark index for the NOSE TIP in the Face Mesh model (index 1)
NOSE_TIP_LANDMARK_INDEX = 1

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        # If loading from a video file, use 'break' instead of 'continue'.
        continue

    # Flip the image horizontally for a selfie-view display and convert BGR to RGB
    image = cv2.flip(image, 1)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Process the image with MediaPipe
    results = face_mesh.process(image_rgb)

    # Image dimensions
    h, w, _ = image.shape
    center_x = w // 2
    
    current_time = time.time()
    face_detected = False

    # --- Anomaly Detection and Drawing ---
    if results.multi_face_landmarks:
        face_detected = True
        
        # We assume only one face (index 0) is being tracked
        face_landmarks = results.multi_face_landmarks[0]
        
        # Get the nose tip coordinates (normalized to 0-1)
        nose_tip = face_landmarks.landmark[NOSE_TIP_LANDMARK_INDEX]
        
        # Convert normalized coordinates to pixel coordinates
        nose_x_px = int(nose_tip.x * w)
        nose_y_px = int(nose_tip.y * h)
        
        # Calculate horizontal deviation from the center
        horizontal_deviation = abs(nose_x_px - center_x)

        # Draw a line from the nose to the center for visualization
        cv2.line(image, (nose_x_px, nose_y_px), (center_x, nose_y_px), (255, 255, 0), 2)
        
        # Check for significant deviation
        if horizontal_deviation > DEVIATION_THRESHOLD_PIXELS:
            # --- Deviation Detected ---
            if not is_deviating:
                # Start tracking time for a new deviation event
                start_time_deviation = current_time
                is_deviating = True
                anomaly_triggered = False
            
            elapsed_time = current_time - start_time_deviation
            
            # Draw visual cues for deviation
            status_text = f"DEVIATION: {elapsed_time:.1f}s"
            status_color = (0, 0, 255) # Red
            
            # Check if the time threshold is exceeded
            if elapsed_time >= ANOMALY_TRIGGER_TIME:
                if not anomaly_triggered:
                    print(f"\n[CRITICAL ANOMALY] Head deviated for >{ANOMALY_TRIGGER_TIME} seconds.")
                    anomaly_triggered = True
                status_text = "ANOMALY TRIGGERED! LOGGED!"
                
            # Draw the nose marker (Red when deviating)
            cv2.circle(image, (nose_x_px, nose_y_px), 5, (0, 0, 255), -1)

        else:
            # --- Within Normal Range ---
            is_deviating = False
            start_time_deviation = None
            anomaly_triggered = False
            
            status_text = "STATUS: Face Centered"
            status_color = (0, 255, 0) # Green
            
            # Draw the nose marker (Green when centered)
            cv2.circle(image, (nose_x_px, nose_y_px), 5, (0, 255, 0), -1)
            
        # Display the status text
        cv2.putText(image, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)

    # --- No Face Detected Anomaly ---
    if not face_detected:
        if not is_deviating:
            # Start tracking time for a "no face" event
            start_time_deviation = current_time
            is_deviating = True
            anomaly_triggered = False
            
        elapsed_time = current_time - start_time_deviation
        
        status_text = f"WARNING: No Face! {elapsed_time:.1f}s"
        status_color = (0, 165, 255) # Orange
        
        if elapsed_time >= ANOMALY_TRIGGER_TIME:
            if not anomaly_triggered:
                print(f"\n[CRITICAL ANOMALY] No face detected for >{ANOMALY_TRIGGER_TIME} seconds.")
                anomaly_triggered = True
            status_text = "ANOMALY TRIGGERED! NO FACE!"
            status_color = (0, 0, 255) # Red

        cv2.putText(image, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2, cv2.LINE_AA)
        
    # Draw the central line and deviation area boundary
    cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 1) # Center line (White)
    cv2.line(image, (center_x - DEVIATION_THRESHOLD_PIXELS, 0), (center_x - DEVIATION_THRESHOLD_PIXELS, h), (255, 0, 0), 1) # Left boundary
    cv2.line(image, (center_x + DEVIATION_THRESHOLD_PIXELS, 0), (center_x + DEVIATION_THRESHOLD_PIXELS, h), (255, 0, 0), 1) # Right boundary


    # Display the final image
    cv2.imshow('Real-Time Proctoring Core', image)

    # Press 'q' to exit the loop
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --- Cleanup ---
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
