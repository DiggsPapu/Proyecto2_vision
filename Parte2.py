import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolov8n.pt")

video_path = 'Danzan3.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize MediaPipe
mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# Initialize holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

if not cap.isOpened():
    print("No se pudo abrir el video")
    exit()

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video: double width
output_path = 'video_doble.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height))

# Output video: only skeleton
skeleton_output_path = 'video_skeleton.mp4'
out_skeleton = cv2.VideoWriter(skeleton_output_path, fourcc, fps, (width, height))


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert frame to RGB for processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLO detection
    yolo_results = model(frame_rgb, verbose=False)

    # Create black frame for skeletons
    skeleton_frame = np.zeros_like(frame)

    # Process each detected person
    for result in yolo_results:
        for box in result.boxes:
            # Si el objeto detectado es una persona (class 0 in YOLO)
            if int(box.cls[0]) == 0:
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Extract the person ROI (Region of Interest)
                person_roi = frame_rgb[y1:y2, x1:x2]

                # Skip if ROI is too small
                if person_roi.size == 0:
                    continue

                # Process the person ROI with MediaPipe
                results = holistic.process(person_roi)


                # Dibujar manualmente el cuerpo (pose) en coordenadas del frame completo
                if results.pose_landmarks:
                    for connection in mp_holistic.POSE_CONNECTIONS:
                        start_idx, end_idx = connection
                        start = results.pose_landmarks.landmark[start_idx]
                        end = results.pose_landmarks.landmark[end_idx]

                        # Reescalar y trasladar coordenadas al frame original
                        x_start = int(start.x * (x2 - x1) + x1)
                        y_start = int(start.y * (y2 - y1) + y1)
                        x_end = int(end.x * (x2 - x1) + x1)
                        y_end = int(end.y * (y2 - y1) + y1)

                        # Dibujar en el frame original
                        cv2.circle(frame, (x_start, y_start), 3, (0, 0, 255), -1)
                        cv2.line(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)

                        # Dibujar en el frame de esqueleto
                        cv2.circle(skeleton_frame, (x_start, y_start), 3, (0,0 , 255), -1)
                        cv2.line(skeleton_frame, (x_start, y_start), (x_end, y_end),(0, 255, 0), 2)


    # Combine frames horizontally
    combined_frame = np.hstack((frame, skeleton_frame))

    # Display and save
    cv2.imshow("Original + Esqueleto", combined_frame)
    out.write(combined_frame)
    out_skeleton.write(skeleton_frame)  # Save skeleton-only frame
    if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
        break

# Release resources
cap.release()
out.release()
out_skeleton.release()
cv2.destroyAllWindows()
