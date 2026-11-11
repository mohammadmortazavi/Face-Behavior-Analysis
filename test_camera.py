import cv2
import time
import os

# --- Settings ---
CAMERA_INDEX = 0          
BACKEND = cv2.CAP_DSHOW   
WINDOW_NAME = "Face & Eye Tracker"
SAVE_DIR = "captures"     

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

def main():
    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    print("Camera opened successfully.")
    print("Key controls:")
    print("  [q] or [ESC] : Exit")
    print("  [s]          : Save Photo manually")
    print("  [e]          : Toggle EYE FOCUS Mode (ON/OFF)")
    print("  [m]          : Toggle AUTO-RECORD on Motion")

    # Load YuNet Face Detector
    model_path = "face_detection_yunet_2023mar.onnx"
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        return
    
    face_detector = cv2.FaceDetectorYN.create(model_path, "", (320, 320), 0.8, 0.3, 5000)

    # متغیرها
    eye_focus_mode = False  # حالت تمرکز روی چشم‌ها به صورت پیش‌فرض خاموش است
    recording = False
    auto_record_on_motion = False
    video_writer = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # --- تشخیص چهره با YuNet ---
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(frame)
        
        if faces is not None:
            for face in faces:
                # مختصات کادر صورت
                box = face[:4].astype(int)
                x, y, w_f, h_f = box
                
                # مختصات چشم‌ها (اندیس‌های 4،5 برای چشم راست و 6،7 برای چشم چپ)
                right_eye = face[4:6].astype(int)
                left_eye = face[6:8].astype(int)
                
                if eye_focus_mode:
                    # --- حالت تمرکز روی چشم ---
                    # کشیدن کادر دور صورت به رنگ آبی و کم‌رنگ
                    cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (255, 0, 0), 1)
                    
                    # کشیدن دایره روی چشم چپ و راست
                    cv2.circle(frame, tuple(left_eye), 8, (0, 255, 255), 2) # زرد
                    cv2.circle(frame, tuple(right_eye), 8, (0, 255, 255), 2) # زرد
                    
                    # نقطه وسط چشم‌ها
                    cv2.circle(frame, tuple(left_eye), 2, (0, 0, 255), -1) # قرمز
                    cv2.circle(frame, tuple(right_eye), 2, (0, 0, 255), -1) # قرمز
                    
                    cv2.putText(frame, "Eye Focus Mode: ON", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                else:
                    # --- حالت عادی (کادر قرمز دور صورت) ---
                    cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), (0, 0, 255), 2)
                    cv2.putText(frame, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)

        # --- کلیدهای میانبر ---
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27: 
            break
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(SAVE_DIR, f"photo_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Photo saved: {filename}")
        elif key == ord('e'):
            # تغییر وضعیت حالت تمرکز روی چشم
            eye_focus_mode = not eye_focus_mode
            status = "ON" if eye_focus_mode else "OFF"
            print(f"Eye Focus Mode is now: {status}")

    cap.release()
    cv2.destroyAllWindows()
    print("Program closed.")

if __name__ == "__main__":
    main()