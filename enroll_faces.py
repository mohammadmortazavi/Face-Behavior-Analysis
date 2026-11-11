import cv2
import numpy as np
import pickle
import os

# --- Settings ---
FACE_MODEL = "face_detection_yunet_2023mar.onnx"
RECOGNITION_MODEL = "face_recognition_sface_2021dec.onnx"
DATABASE_FILE = "known_faces.pkl"

def enroll_person(name, image_path):
    # بارگذاری مدل‌ها
    face_detector = cv2.FaceDetectorYN.create(FACE_MODEL, "", (320, 320), 0.8, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(RECOGNITION_MODEL, "")

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image {image_path}")
        return

    h, w = img.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(img)

    if faces is None:
        print(f"No face detected in {image_path}")
        return

    # فرض می‌کنیم اولین چهره تشخیص داده شده، فرد مورد نظر است
    face = faces[0]
    
    # تراز کردن و برش چهره
    aligned_face = recognizer.alignCrop(img, face)
    
    # استخراج ویژگی (Feature)
    feature = recognizer.feature(aligned_face)
    
    # ذخیره در دیتابیس (دیکشنری)
    database = {}
    if os.path.exists(DATABASE_FILE):
        with open(DATABASE_FILE, 'rb') as f:
            database = pickle.load(f)
    
    database[name] = feature
    
    with open(DATABASE_FILE, 'wb') as f:
        pickle.dump(database, f)
    
    print(f"Successfully enrolled '{name}' and saved to {DATABASE_FILE}")

# --- اجرا ---
# نام شخص و مسیر عکس او را وارد کنید
if __name__ == "__main__":
    # مثال: ثبت‌نام شخصی به نام "Ali" با عکس "ali.jpg"
    # enroll_person("Ali", "ali.jpg")
    
    # برای ثبت‌نام شخصی که همین الان عکس گرفته‌اید:
    print("Enrolling from 'my_face.jpg' as 'Mortazavi'...")
    enroll_person("Mortazavi", "my_face.jpg") 