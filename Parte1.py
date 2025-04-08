import cv2
import mediapipe as mp

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

# Initialize holistic model
holistic = mp_holistic.Holistic(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

# Try camera indices 0 to 2
for i in range(3):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera found at index {i}")
        break
else:
    print("No camera found")
    exit()

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    # Draw landmarks
    #'''
    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Obtener dimensiones del cuadro
        h, w, _ = frame.shape

        # Obtener coordenadas en píxeles de la deteccion
        x_values = [int(lm.x * w) for lm in results.pose_landmarks.landmark]
        y_values = [int(lm.y * h) for lm in results.pose_landmarks.landmark]

        # Calcular bounding box
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # Dibujar rectángulo alrededor del cuerpo
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
     # '''
    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Obtener dimensiones del cuadro
        h, w, _ = frame.shape

        # Obtener coordenadas en píxeles de la deteccion
        x_values = [int(lm.x * w) for lm in results.left_hand_landmarks.landmark]
        y_values = [int(lm.y * h) for lm in results.left_hand_landmarks.landmark]

        # Calcular bounding box
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # Dibujar rectángulo alrededor del objeto detectado
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
     # '''
    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        # Obtener dimensiones del cuadro
        h, w, _ = frame.shape

        # Obtener coordenadas en píxeles de la deteccion
        x_values = [int(lm.x * w) for lm in results.right_hand_landmarks.landmark]
        y_values = [int(lm.y * h) for lm in results.right_hand_landmarks.landmark]

        # Calcular bounding box
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)

        # Dibujar rectángulo alrededor del objeto detectado
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
     # '''
    if results.face_landmarks:
       mp_draw.draw_landmarks(frame, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
       # Obtener dimensiones del cuadro
       h, w, _ = frame.shape

       # Obtener coordenadas en píxeles de la deteccion
       x_values = [int(lm.x * w) for lm in results.face_landmarks.landmark]
       y_values = [int(lm.y * h) for lm in results.face_landmarks.landmark]

       # Calcular bounding box
       x_min, x_max = min(x_values), max(x_values)
       y_min, y_max = min(y_values), max(y_values)

       # Dibujar rectángulo alrededor del objeto detectado
       cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
    #''
    cv2.imshow("Holistic Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27: # esc to exit
        break

cap.release()
cv2.destroyAllWindows()
