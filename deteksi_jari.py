import cv2
import mediapipe as mp

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

# Buka kamera
cap = cv2.VideoCapture(0)

def detect_gesture(hand_landmarks):
    # Misalkan kita hanya mengecek apakah tangan terbuka
    thumb_up = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y
    if thumb_up:
        return 'like'
    return 'unknown'

def count_fingers(hand_landmarks):
    # Posisi referensi jari pada landmark
    finger_tips = [
        mp_hands.HandLandmark.THUMB_TIP,
        mp_hands.HandLandmark.INDEX_FINGER_TIP,
        mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
        mp_hands.HandLandmark.RING_FINGER_TIP,
        mp_hands.HandLandmark.PINKY_TIP
    ]
    
    count = 0
    for finger_tip in finger_tips:
        # Mengecek apakah jari terangkat
        if hand_landmarks.landmark[finger_tip].y < hand_landmarks.landmark[finger_tip - 2].y:
            count += 1
    return count

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Konversi gambar ke RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi tangan
    result = hands.process(rgb_frame)

    # Jika ada tangan yang terdeteksi
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Gambar titik referensi di tangan
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Deteksi gesture
            gesture = detect_gesture(hand_landmarks)
            if gesture == 'like':
                cv2.putText(frame, 'Like Gesture Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Hitung jumlah jari yang terangkat
            num_fingers = count_fingers(hand_landmarks)
            cv2.putText(frame, f'Fingers Raised: {num_fingers}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Membalikkan frame untuk menghilangkan efek mirror
    

    # Tampilkan frame
    cv2.imshow('Deteksi Jari dengan AI', frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
