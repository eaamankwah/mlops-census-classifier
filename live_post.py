"""
Script to POST a sample record to the live deployed API on Heroku (or Render).
Update HEROKU_URL to match your deployed app URL before running.

Usage:
    python live_post.py
"""
import requests

# ── Update this to your actual Heroku/Render app URL ──────────────────────────
HEROKU_URL = "https://mlops-census-eaamankwah-NAME.herokuapp.com"
# ──────────────────────────────────────────────────────────────────────────────

PREDICT_ENDPOINT = f"{HEROKU_URL}/predict"

sample_data = {
    "age": 39,
    "workclass": "State-gov",
    "fnlgt": 77516,
    "education": "Bachelors",
    "education-num": 13,
    "marital-status": "Never-married",
    "occupation": "Adm-clerical",
    "relationship": "Not-in-family",
    "race": "White",
    "sex": "Male",
    "capital-gain": 2174,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

response = requests.post(PREDICT_ENDPOINT, json=sample_data)

print(f"Status code : {response.status_code}")
print(f"Response    : {response.json()}")
