"""
Script to POST a sample record to the live deployed API on Heroku (or Render).
Update HEROKU_URL to match your deployed app URL before running.

Usage:
    python live_post.py
"""
import requests

# ── Update this to your actual Heroku/Render app URL ──────────────────────────
HEROKU_URL = "https://mlops-census-eaamankwah-8cb731658ffd.herokuapp.com"
# ──────────────────────────────────────────────────────────────────────────────

TOKEN_ENDPOINT = f"{HEROKU_URL}/token"
PREDICT_ENDPOINT = f"{HEROKU_URL}/predict"

auth_data = {"username": "alice", "password": "secret"}

print("Requesting access token...")
token_response = requests.post(TOKEN_ENDPOINT, data=auth_data)
print(f"Token status: {token_response.status_code}")

try:
    token_response.raise_for_status()
except requests.RequestException as exc:
    print("Failed to get token:", exc)
    print(token_response.text)
    raise

access_token = token_response.json()["access_token"]
headers = {"Authorization": f"Bearer {access_token}"}

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

response = requests.post(PREDICT_ENDPOINT, json=sample_data, headers=headers)
print(f"Status code : {response.status_code}")
try:
    print(f"Response    : {response.json()}")
except ValueError:
    print("Response is not valid JSON:")
    print(response.text)
