# Topic Classifier

طبقه‌بندی موضوع پست‌های فارسی ویرگول با سنتروید. امبدینگ از API (آدرس در `.env`).

## پایپ‌لاین

1. **بازیابی** — عنوان و بدنه تمیز می‌شوند (ایموجی حذف نمی‌شود)، با مدل امبدینگ `.env` بردار می‌شوند و با ۴۱ سنتروید کسینوس می‌خورند؛ Top-5 برمی‌گردد.
2. **انتخاب** — همیشه Top-1؛ تاپیک ۲ و ۳ فقط اگر فاصله‌شان از Top-1 کمتر از `0.008` باشد.

لایهٔ اعتبار (اسپم) و تأیید LLM در این سرویس نیستند.

## اجرا

در `.env` این‌ها را بگذار (گیت نمی‌شود):

```
EMBEDDING_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=bge-m3
```

مدل باید همان باشد که `data/centroids.npz` با آن ساخته شده.

از ریشهٔ پروژه:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn src.main:app --reload
```

- فرم ساده: `http://127.0.0.1:8000/`
- `POST /classify` با `{title, body, post_id}` — مستندات: `http://127.0.0.1:8000/docs`
- دمو ترمینال: `python scripts/demo_terminal.py` (بدنه را با خط `end` تمام کن)

## دادهٔ لازم سرویس

- `data/centroids.npz`
