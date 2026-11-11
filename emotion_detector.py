import cv2
import os
import numpy as np
import csv
from datetime import datetime
import time
import pickle
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import simpledialog, messagebox

# --- Settings ---
CAMERA_INDEX = 0          
BACKEND = cv2.CAP_DSHOW   
WINDOW_NAME = "Behavior Analysis - Face, Emotion, Pose & Identity"
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
EMOTION_MODEL = "emotion-ferplus-8.onnx"
POSE_MODEL = "pose_landmarker_lite.task"
RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"
DATABASE_FILE = "known_faces.pkl"
LOG_FILE = "behavior_logs.csv"

CONFIDENCE_THRESHOLD = 0.50
LOG_INTERVAL_SECONDS = 3.0
RECOGNITION_THRESHOLD = 0.363 

def init_log_file():
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "Identity", "Emotion", "Confidence", "Pose_Alert"])

def log_behavior(identity, emotion, confidence, alert=""):
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, identity, emotion, f"{confidence*100:.1f}", alert])

def enroll_new_face(face_img, face_coords, recognizer):
    """گرفتن نام از کاربر و ثبت چهره در دیتابیس"""
    # مخفی کردن پنجره اصلی Tkinter
    root = tk.Tk()
    root.withdraw() 
    
    # پرسیدن نام شخص
    name = simpledialog.askstring("ثبت‌نام چهره جدید", "لطفاً نام شخص را وارد کنید:")
    
    if not name:
        print("Enrollment cancelled.")
        return False

    try:
        # استخراج ویژگی از چهره
        aligned_face = recognizer.alignCrop(face_img, face_coords)
        feature = recognizer.feature(aligned_face)
        
        # بارگذاری دیتابیس فعلی
        database = {}
        if os.path.exists(DATABASE_FILE):
            with open(DATABASE_FILE, 'rb') as f:
                database = pickle.load(f)
        
        # اضافه کردن شخص جدید
        database[name] = feature
        
        # ذخیره مجدد دیتابیس
        with open(DATABASE_FILE, 'wb') as f:
            pickle.dump(database, f)
            
        messagebox.showinfo("موفقیت", f"چهره '{name}' با موفقیت در دیتابیس ذخیره شد.")
        print(f"[ENROLLED] {name} added to database.")
        return True
    except Exception as e:
        messagebox.showerror("خطا", f"خطا در ثبت‌نام: {str(e)}")
        print(f"Error during enrollment: {e}")
        return False

def main():
    init_log_file()

    cap = cv2.VideoCapture(CAMERA_INDEX, BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    required_files = [FACE_MODEL, EMOTION_MODEL, POSE_MODEL, RECOGNITION_MODEL]
    if not all(os.path.exists(f) for f in required_files):
        print("Error: One or more model files are missing!")
        return

    face_detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320), 0.8, 0.3, 5000)
    emotion_net = cv2.dnn.readNetFromONNX(EMOTION_MODEL)
    EMOTIONS = ["Neutral", "Happy", "Surprise", "Sad", "Angry", "Disgust", "Fear", "Contempt"]
    
    base_options = python.BaseOptions(model_asset_path=POSE_MODEL)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1
    )
    pose_landmarker = vision.PoseLandmarker.create_from_options(options)

    recognizer = cv2.FaceRecognizerSF.create(RECOGNITION_MODEL, "")
    
    known_faces = {}
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            known_faces = pickle.load(f)
        print(f"Loaded {len(known_faces)} known faces from database.")

    print("Program started. Press 'q' to exit.")
    print("Press 'n' to Enroll a NEW face.")
    
    last_logged_emotion = ""
    last_log_time = 0
    frame_timestamp_ms = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        frame_timestamp_ms += 33 
        
        # --- تشخیص چهره، احساسات و هویت ---
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(frame)
        
        dominant_emotion = "Neutral"
        score = 0.0
        pose_alert = ""
        current_face_coords = None # برای ذخیره مختصات چهره در صورت فشردن کلید N
        current_face_img = None
        
        if faces is not None:
            # فرض می‌کنیم فقط یک چهره اصلی در کادر داریم
            face = faces[0]
            x, y, w_f, h_f = face[:4].astype(int)
            current_face_coords = face
            current_face_img = frame.copy() # کپی از فریم برای ثبت‌نام
            
            pad = 20
            y1, y2 = max(0, y - pad), min(h, y + h_f + pad)
            x1, x2 = max(0, x - pad), min(w, x + w_f + pad)
            face_img = frame[y1:y2, x1:x2]
            
            # --- تشخیص احساسات ---
            if face_img.size != 0:
                try:
                    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                    resized_face = cv2.resize(gray_face, (64, 64))
                    blob = cv2.dnn.blobFromImage(resized_face, 1.0, (64, 64), (0, 0, 0), swapRB=False)
                    emotion_net.setInput(blob)
                    preds = emotion_net.forward()[0]
                    
                    emotion_idx = np.argmax(preds)
                    dominant_emotion = EMOTIONS[emotion_idx]
                    score = preds[emotion_idx]
                except Exception:
                    pass

            # --- تشخیص هویت ---
            identity = "Unknown"
            if known_faces:
                try:
                    aligned_face = recognizer.alignCrop(frame, face)
                    feature = recognizer.feature(aligned_face)
                    
                    best_score = -1
                    best_name = "Unknown"
                    
                    for name, known_feature in known_faces.items():
                        match_score = recognizer.match(feature, known_feature, cv2.FaceRecognizerSF_FR_COSINE)
                        if match_score > best_score:
                            best_score = match_score
                            best_name = name
                    
                    if best_score >= RECOGNITION_THRESHOLD:
                        identity = best_name
                except Exception:
                    pass

            # --- نمایش اطلاعات روی تصویر ---
            color = (0, 255, 0) if dominant_emotion == "Happy" else (0, 0, 255) if dominant_emotion == "Angry" else (200, 200, 200)
            if identity != "Unknown":
                color = (255, 255, 0)

            cv2.rectangle(frame, (x, y), (x + w_f, y + h_f), color, 2)
            label = f"{identity} | {dominant_emotion} ({score*100:.1f}%)"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - text_h - 10), (x + text_w, y), color, -1)
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # --- ژست بدن ---
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        pose_result = pose_landmarker.detect_for_video(mp_image, frame_timestamp_ms)
        
        if pose_result.pose_landmarks:
            landmarks = pose_result.pose_landmarks[0]
            for connection in vision.PoseLandmarksConnections.POSE_LANDMARKS:
                start_px = (int(landmarks[connection.start].x * w), int(landmarks[connection.start].y * h))
                end_px = (int(landmarks[connection.end].x * w), int(landmarks[connection.end].y * h))
                cv2.line(frame, start_px, end_px, (0, 255, 0), 2)
            
            for landmark in landmarks:
                cv2.circle(frame, (int(landmark.x * w), int(landmark.y * h)), 4, (0, 0, 255), -1)
            
            # بررسی پرخاشگری
            hands_raised = (landmarks[15].y < landmarks[11].y) or (landmarks[16].y < landmarks[12].y)
            if dominant_emotion == "Angry" and score > 0.6 and hands_raised:
                pose_alert = "AGGRESSIVE POSE DETECTED"
                cv2.putText(frame, pose_alert, (w//2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 5)

        # --- لاگ‌گیری ---
        current_time = time.time()
        if (score >= CONFIDENCE_THRESHOLD and 
            (dominant_emotion != last_logged_emotion or current_time - last_log_time > LOG_INTERVAL_SECONDS)):
            # فقط اگر شخص شناسایی شده باشد لاگ می‌کنیم (برای جلوگیری از پر شدن فایل با Unknown)
            if 'identity' in locals() and identity != "Unknown":
                log_behavior(identity, dominant_emotion, score, pose_alert)
                last_logged_emotion = dominant_emotion
                last_log_time = current_time

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            # اگر چهره‌ای در کادر باشد، فرآیند ثبت‌نام شروع می‌شود
            if current_face_img is not None and current_face_coords is not None:
                success = enroll_new_face(current_face_img, current_face_coords, recognizer)
                if success:
                    # بارگذاری مجدد دیتابیس برای اعمال تغییرات
                    with open(DATABASE_FILE, 'rb') as f:
                        known_faces = pickle.load(f)
            else:
                print("No face detected to enroll. Please look at the camera.")

    cap.release()
    pose_landmarker.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()