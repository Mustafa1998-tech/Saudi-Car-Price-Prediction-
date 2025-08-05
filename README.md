# توقع أسعار السيارات في السعودية (Saudi Car Price Prediction)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

نموذج ذكاء اصطناعي للتنبؤ بأسعار السيارات المستعملة في السوق السعودي بناءً على مواصفاتها.

## المميزات

- جمع وتنظيف بيانات السيارات من السوق السعودي
- تحليل البيانات وعرض الإحصائيات والرسوم البيانية
- تدريب نماذج تعلم آلي متعددة للتنبؤ بالسعر
- واجهة ويب تفاعلية سهلة الاستخدام
- دعم كامل للغة العربية
- نشر سهل على منصات مختلفة (Docker, Heroku, Render, Hugging Face)

## المتطلبات

- Python 3.8 أو أحدث
- pip (مدير حزم بايثون)
- Docker (اختياري للنشر باستخدام الحاويات)

## التثبيت

### التثبيت المحلي

1. استنسخ المستودع:
   ```bash
   git clone https://github.com/yourusername/saudi-car-price-prediction.git
   cd saudi-car-price-prediction
   ```

2. قم بإنشاء بيئة افتراضية جديدة (اختياري لكن موصى به):
   ```bash
   python -m venv venv
   source venv/bin/activate  # لنظام Linux/Mac
   .\venv\Scripts\activate  # لنظام Windows
   ```

3. قم بتثبيت المتطلبات:
   ```bash
   pip install -r requirements.txt
   ```

### التثبيت باستخدام Docker

1. تأكد من تثبيت Docker و Docker Compose على جهازك.

2. قم ببناء وتشغيل التطبيق:
   ```bash
   docker-compose up --build
   ```

3. افتح المتصفح على العنوان:
   ```
   http://localhost:8501
   ```

## طريقة الاستخدام

### 1. تنظيف البيانات

قم بتشغيل البرنامج النصي لتنظيف البيانات:
```bash
python data_cleaning.py
```

### 2. تحليل البيانات الاستكشافي (اختياري)

قم بإنشاء رسوم بيانية وتحليلات للبيانات:
```bash
python eda.py
```

### 3. تدريب النموذج

قم بتدريب النموذج وحفظه:
```bash
python train_model.py
```

### 4. تشغيل التطبيق

لتشغيل التطبيق محلياً:
```bash
python deploy.py
```

### 5. إجراء تنبؤات

لإجراء تنبؤات من سطر الأوامر:
```bash
python inference.py --brand "تويوتا" --model "كامري" --year 2020 --kilometers 50000 --fuel_type "بنزين" --gear_type "أوتوماتيك" --car_condition "جيدة"
```

## النشر

### النشر على Render

1. أنشئ حساباً على [Render](https://render.com/)
2. انقر على "New" واختر "Web Service"
3. قم بربط حساب GitHub الخاص بك واختر المستودع
4. استخدم الإعدادات التالية:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
5. اضغط على "Deploy"

### النشر على Hugging Face Spaces

1. أنشئ حساباً على [Hugging Face](https://huggingface.co/)
2. أنشئ مساحة جديدة (New Space)
3. اختر "Docker" كنوع المساحة
4. انسخ محتويات المستودع إلى المساحة
5. أضف ملف `Dockerfile` و `requirements.txt`
6. اضغط على "Create Space"

## هيكل المشروع

```
saudi-car-price-prediction/
├── data/
│   ├── raw/               # البيانات الأولية
│   └── processed/         # البيانات المعالجة
├── models/                # النماذج المدربة والمشفرة
├── reports/               # التقارير والنتائج
├── eda/                   # تحليلات البيانات الاستكشافية
├── .env                  # متغيرات البيئة
├── .gitignore
├── app.py                # تطبيق Streamlit
├── data_cleaning.py      # تنظيف البيانات
├── debug.py              # تصحيح الأخطاء
├── deploy.py             # نشر التطبيق
├── docker-compose.yml    # تكوين Docker Compose
├── Dockerfile            # تكوين Docker
├── eda.py                # تحليل البيانات الاستكشافي
├── inference.py          # التنبؤ من سطر الأوامر
├── Procfile              # تكوين النشر
├── README.md             # هذا الملف
├── requirements.txt      # التبعيات
├── runtime.txt           # إصدار Python
├── saudi_cars.csv        # بيانات السيارات
├── setup.sh              # إعداد النظام
└── train_model.py        # تدريب النموذج
```

## المساهمة

المساهمات مرحب بها! يمكنك المساعدة من خلال:

1. إضافة المزيد من البيانات
2. تحسين النموذج
3. تحسين واجهة المستخدم
4. الإبلاغ عن الأخطاء

## الترخيص

هذا المشروع مرخص تحت [MIT License](LICENSE).

## التواصل

للاستفسارات أو الاقتراحات، يرجى فتح "Issue" جديد في المستودع.

---

تم التطوير بواسطة [اسمك] - 2024
