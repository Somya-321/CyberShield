# 🛡️ CyberShield: Phishing Detection

A simple and effective machine learning project that detects **phishing URLs** and **phishing emails** using trained models.

---

## ⚙️ What It Does

- 🔗 **URL Detection**: Checks if a given website URL is suspicious or safe.
- 📧 **Email Detection**: Identifies whether an email message is legitimate or a phishing attempt.

---

## 💻 How to Use

1. 🔁 Train models using `model.py`
2. 🌐 Run the app using `app.py`
3. 📝 Enter a URL or email content in the web interface
4. ✅ Get instant results on whether it's phishing or safe

---

## 📂 Files

- `model.py` – Trains and saves the models
- `app.py` – Flask web app for prediction
- `templates/` – HTML pages for input and results

---

## ✅ Requirements

- Python
- scikit-learn
- Flask
- pandas
- joblib

Install with:

```bash
pip install -r requirements.txt
