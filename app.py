from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)

# Load the trained models
url_model = joblib.load('phishing_model.pkl')
email_model = joblib.load('email_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')


# Function to extract URL features (same as used in model.py for training)
def extract_url_features(url):
    features = []

    # 1. Having IP Address
    ip_pattern = re.compile(
        r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$')
    features.append(-1 if ip_pattern.match(url) else 1)

    # 2. URL Length
    features.append(1 if len(url) < 54 else 0 if len(url) <= 75 else -1)

    # 3. Shortening Service
    shortening_services = ['bit.ly', 'goo.gl', 'tinyurl.com', 't.co']
    features.append(-1 if any(service in url.lower() for service in shortening_services) else 1)

    # 4. Having @ Symbol
    features.append(-1 if '@' in url else 1)

    # 5. Double Slash Redirecting
    features.append(-1 if '//' in url[7:] else 1)

    # 6. Prefix Suffix
    features.append(-1 if '-' in url.split('.')[0] else 1)

    # 7. Having Sub Domain
    domain = url.split('//')[-1].split('/')[0]
    subdomains = domain.split('.')
    features.append(1 if len(subdomains) <= 2 else 0 if len(subdomains) == 3 else -1)

    # 8. SSLfinal_State (simplified check for HTTPS)
    features.append(1 if url.startswith('https') else -1)

    # 9. Domain_registration_length (placeholder)
    features.append(1)

    # 10. Favicon (placeholder)
    features.append(-1)

    # 11. Port (placeholder)
    features.append(1)

    # 12. HTTPS_token
    features.append(-1 if 'https' in url.lower().split('//')[-1] else 1)

    # 13. Request_URL (placeholder)
    features.append(0)

    # 14. URL_of_Anchor (placeholder)
    features.append(0)

    # 15. Links_in_tags (placeholder)
    features.append(0)

    # 16. SFH (placeholder)
    features.append(0)

    # 17. Submitting_to_email
    features.append(-1 if 'mail' in url.lower() or 'email' in url.lower() else 1)

    # 18. Abnormal_URL (placeholder)
    features.append(0)

    # 19. Redirect (placeholder)
    features.append(0)

    # 20. on_mouseover (placeholder)
    features.append(1)

    # 21. RightClick (placeholder)
    features.append(1)

    # 22. popUpWindow (placeholder)
    features.append(1)

    # 23. Iframe (placeholder)
    features.append(1)

    # 24. age_of_domain (placeholder)
    features.append(1)

    # 25. DNSRecord (placeholder)
    features.append(1)

    # 26. web_traffic (placeholder)
    features.append(1)

    # 27. Page_Rank (placeholder)
    features.append(1)

    # 28. Google_Index (placeholder)
    features.append(1)

    # 29. Links_pointing_to_page (placeholder)
    features.append(1)

    # 30. Statistical_report (placeholder)
    features.append(1)

    return features


# Home route
@app.route('/')
def index():
    return render_template('index.html')


# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    url = request.form.get('url')
    email = request.form.get('email')

    url_result = None
    email_result = None

    # Predict URL phishing
    if url:
        url_features = extract_url_features(url)
        url_features = np.array(url_features).reshape(1, -1)
        url_pred = url_model.predict(url_features)[0]
        url_result = 'Phishing' if url_pred == 0 else 'Legitimate'

    # Predict email phishing
    if email:
        email_transformed = tfidf_vectorizer.transform([email])
        email_pred = email_model.predict(email_transformed)[0]
        email_result = 'Phishing' if email_pred == 'Phishing Email' else 'Safe'

    return render_template('result.html', url_result=url_result, email_result=email_result)


if __name__ == '__main__':
    app.run(debug=True)