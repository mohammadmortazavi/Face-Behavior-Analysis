# Face & Behavior Analysis System

یک سیستم هوشمند تحلیل رفتار و کنترل دست# Face & Behavior Analysis System

یک سیستم هوشمند تحلیل رفتار و کنترل دسترسی با استفاده از وب‌کم، پایتون و OpenCV.

## ویژگی‌ها
- تشخیص چهره و احساسات (FERPlus)
- تشخیص ژست بدن (MediaPipe)
- تشخیص هویت و ثبت‌نام چهره (SFace)
- ذخیره لاگ رفتار در فایل CSV

## پیش‌نیازها
- Python 3.11 (توصیه می‌شود)
- دوربین وب‌کم متصل به سیستم

## راه‌اندازی محیط (نصب)

برای اجرای این پروژه، نباید از محیط پایتون اصلی سیستم خود استفاده کنید. مراحل زیر را دقیقاً دنبال کنید تا یک محیط مجازی ایزوله بسازید:

1. **دریافت کدها**:
   ```bash
   git clone https://github.com/mohammadmortazavi/Face-Behavior-Analysis.git
   cd Face-Behavior-Analysis

- ## دانلود مدل‌ها
فایل‌های زیر را دانلود کرده و در پوشه اصلی پروژه (کنار `emotion_detector.py`) قرار دهید:

1. **مدل تشخیص چهره (YuNet):** [دانلود](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
2. **مدل تشخیص احساسات (FERPlus):** [دانلود](https://github.com/onnx/models/raw/main/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx)
3. **مدل تشخیص ژست (MediaPipe):** [دانلود](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task)
4. **مدل تشخیص هویت (SFace):** [دانلود](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx)

## نحوه اجرا
`python emotion_detector.py`


ساخت محیط مجازی
python -m venv venv_tf
فعال‌سازی محیط مجازی
venv_tf\Scripts\activate
نصب کتابخانه‌های مورد نیاز
pip install -r requirements.txt
افایل‌های مدل به دلیل حجم بالا در مخزن گیت‌هاب قرار داده نشده‌اند. لطفاً فایل‌های زیر را دانلود کرده و در پوشه اصلی پروژه (کنار emotion_detector.py) قرار دهید:

دستورات کلیدی
q: خروج از برنامه

n: ثبت‌نام چهره جدید (در حین اجرای دوربین)
