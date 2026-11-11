# Face & Behavior Analysis System

یک سیستم هوشمند تحلیل رفتار و کنترل دسترسی با استفاده از وبکم، پایتون و OpenCV.

## ویژگی‌ها
- تشخیص چهره و احساسات (FERPlus)
- تشخیص ژست بدن (MediaPipe)
- تشخیص هویت و ثبت‌نام چهره (SFace)
- ذخیره لاگ رفتار در فایل CSV

## پیش‌نیازها
- Python 3.11
- نصب کتابخانه‌ها: `pip install -r requirements.txt`

## دانلود مدل‌ها
فایل‌های زیر را دانلود کرده و در پوشه اصلی پروژه قرار دهید:
- `face_detection_yunet_2023mar.onnx`
- `emotion-ferplus-8.onnx`
- `pose_landmarker_lite.task`
- `face_recognition_sface_2021dec.onnx`

## نحوه اجرا
`python emotion_detector.py`