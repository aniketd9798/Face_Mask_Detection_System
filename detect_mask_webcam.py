import cv2
import numpy as np
from tensorflow.keras.models import load_model

print("[INFO] Loading AI brain and face detector...")
# 1. Load your newly trained AI model
model = load_model("mask_detector.h5")

# 2. Load the OpenCV Face Detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 3. Start the Webcam
cap = cv2.VideoCapture(0)

print("[INFO] Starting video stream. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale for the face detector to work
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Find all faces in the current frame
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

    # Loop through every face detected
    for (x, y, w, h) in faces:
        # 4. Cut out just the face from the video
        face_crop = frame[y:y+h, x:x+w]
        
        # Safety check to prevent resizing errors
        if face_crop.shape[0] == 0 or face_crop.shape[1] == 0:
            continue

        # 5. Prepare the face image for the AI
        face_crop = cv2.resize(face_crop, (224, 224))
        face_crop = face_crop / 255.0
        face_crop = np.expand_dims(face_crop, axis=0) # Add a batch dimension

        # 6. Ask the AI for a prediction
        prediction = model.predict(face_crop, verbose=0)[0]
        
        if prediction[0] > prediction[1]:
            label = "Mask"
            color = (0, 255, 0)  # Green for Mask
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red for No Mask

        # 7. Draw the bounding box and the label on the screen
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the final video frame
    cv2.imshow("Real-Time Face Mask Detector", frame)

    # Quit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()