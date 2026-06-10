"""
FastAPI application for Census Income inference.
Exposes:
  GET  /            — welcome message (public)
  POST /token       — login to get a JWT access token
  GET  /users/me    — get current authenticated user info
  POST /predict     — model inference (requires Bearer token)
"""
import os
import pickle
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Heroku: pull model artifacts from DVC on first startup
# ---------------------------------------------------------------------------
if "DYNO" in os.environ and os.path.isdir(".dvc"):
    os.system("dvc config core.no_scm true")
    if os.system("dvc pull") != 0:
        exit("dvc pull failed")
    os.system("rm -r .dvc .apt/usr/lib/dvc")

# ---------------------------------------------------------------------------
# Load model artifacts
# ---------------------------------------------------------------------------
_BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE, "model", "model.pkl")
ENCODER_PATH = os.path.join(_BASE, "model", "encoder.pkl")
LB_PATH = os.path.join(_BASE, "model", "lb.pkl")
# Lazy-loaded model artifacts (avoid heavy imports/load at import-time on Heroku)
model = None
encoder = None
lb = None


def _load_artifacts() -> None:
    """Load model, encoder and label binarizer into module globals."""
    global model, encoder, lb
    if model is not None and encoder is not None and lb is not None:
        return
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    with open(LB_PATH, "rb") as f:
        lb = pickle.load(f)

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# ---------------------------------------------------------------------------
# JWT / Auth configuration
# ---------------------------------------------------------------------------
SECRET_KEY = os.environ.get(
    "SECRET_KEY",
    "09d25e094faa6ca2556c818166b7a9563b93f7099f6f0f4caa6cf63b88e8d3e7",
)
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["sha256_crypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---------------------------------------------------------------------------
# In-memory user "database"  (swap for a real DB in production)
# Passwords below hash to "secret" (alice) and "password" (bob).
# ---------------------------------------------------------------------------
FAKE_USERS_DB: dict = {
    "alice": {
        "username": "alice",
        "full_name": "Alice Demo",
        "email": "alice@example.com",
        "hashed_password": (
            "$5$rounds=535000$b1kLcPhjCiwghqcJ"
            "$P/FZP.GkztjK4kG63XNWpbHQ6SncIfs2n/ZCrCy8cN0"
        ),
        "disabled": False,
    },
    "bob": {
        "username": "bob",
        "full_name": "Bob Inactive",
        "email": "bob@example.com",
        "hashed_password": (
            "$5$rounds=535000$AAoiuwqvPBXAXkNj"
            "$azNM.9jfDC1qBZwPRyJBhDj3abM46LULmg.fnc7z/Q."
        ),
        "disabled": True,
    },
}


# ---------------------------------------------------------------------------
# Auth schemas & helpers
# ---------------------------------------------------------------------------
class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None


class UserInDB(User):
    hashed_password: str


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_user(db: dict, username: str) -> Optional[UserInDB]:
    if username in db:
        return UserInDB(**db[username])
    return None


def authenticate_user(db: dict, username: str, password: str) -> Optional[UserInDB]:
    user = get_user(db, username)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (
        expires_delta if expires_delta else timedelta(minutes=15)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserInDB:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(FAKE_USERS_DB, token_data.username)
    if user is None:
        raise credentials_exception
    return user


async def get_current_active_user(
    current_user: UserInDB = Depends(get_current_user),
) -> UserInDB:
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ---------------------------------------------------------------------------
# Pydantic inference schema
# ---------------------------------------------------------------------------
class CensusInput(BaseModel):
    age: int = Field(...)
    workclass: str = Field(...)
    fnlgt: int = Field(...)
    education: str = Field(...)
    education_num: int = Field(..., alias="education-num")
    marital_status: str = Field(..., alias="marital-status")
    occupation: str = Field(...)
    relationship: str = Field(...)
    race: str = Field(...)
    sex: str = Field(...)
    capital_gain: int = Field(..., alias="capital-gain")
    capital_loss: int = Field(..., alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week")
    native_country: str = Field(..., alias="native-country")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
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
        },
    }


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Census Income Prediction API",
    description=(
        "Predicts whether a person earns >50K or <=50K per year "
        "based on US Census Bureau data.\n\n"
        "## 🔐 Authentication\n\n"
        "Protected endpoints require a **Bearer JWT token**.\n\n"
        "**Step 1** — Click the `/token` endpoint below, then **Try it out**.\n\n"
        "**Step 2** — Enter `username=alice` and `password=secret` and Execute.\n\n"
        "**Step 3** — Copy the `access_token` value from the response.\n\n"
        "**Step 4** — Click the **Authorize 🔒** button at the top of this page, "
        "paste the token, and click Authorize.\n\n"
        "All `/predict` and `/users/me` calls will now include your token "
        "automatically.\n\n"
        "| Credential | Value |\n"
        "|---|---|\n"
        "| username | `alice` |\n"
        "| password | `secret` |\n"
    ),
    version="2.0.0",
)

# ---------------------------------------------------------------------------
# CORS Configuration  (ADDED LAYER)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins, required if testing via file:// protocol
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> dict:
    """Public welcome endpoint — no authentication required."""
    return {
        "message": "Welcome to the Census Income Prediction API!",
        "version": "2.0.0",
        "docs": "/docs",
        "authentication": {
            "instructions": (
                "POST /token with form fields username and password "
                "to receive a Bearer token, then include it as "
                "Authorization: Bearer <token> on POST /predict"
            ),
            "demo_username": "alice",
            "demo_password": "secret",
            "token_endpoint": "/token",
        },
        "quick_start": (
            "Step 1: POST /token  body: username=alice&password=secret  "
            "Step 2: copy access_token  "
            "Step 3: POST /predict  header: Authorization: Bearer <token>"
        ),
    }


@app.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
) -> Token:
    """
    Authenticate with username + password and receive a JWT Bearer token.

    Demo credentials:  username=alice  password=secret
    """
    user = authenticate_user(FAKE_USERS_DB, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )
    return Token(access_token=access_token, token_type="bearer")


@app.get("/users/me", response_model=User)
async def read_users_me(
    current_user: UserInDB = Depends(get_current_active_user),
) -> User:
    """Return the currently authenticated user's profile."""
    return current_user


@app.post("/predict")
async def predict(
    data: CensusInput,
    current_user: UserInDB = Depends(get_current_active_user),
) -> dict:
    """
    Run model inference on a single Census record (authentication required).

    Returns the predicted salary class: '>50K' or '<=50K'.
    """
    # Ensure model artifacts are loaded (lazy-load to save memory on startup)
    _load_artifacts()

    row = {
        "age": data.age,
        "workclass": data.workclass,
        "fnlgt": data.fnlgt,
        "education": data.education,
        "education-num": data.education_num,
        "marital-status": data.marital_status,
        "occupation": data.occupation,
        "relationship": data.relationship,
        "race": data.race,
        "sex": data.sex,
        "capital-gain": data.capital_gain,
        "capital-loss": data.capital_loss,
        "hours-per-week": data.hours_per_week,
        "native-country": data.native_country,
    }
    df = pd.DataFrame([row])

    X_cat = encoder.transform(df[CAT_FEATURES].values)
    X_cont = df.drop(CAT_FEATURES, axis=1).values
    X = np.concatenate([X_cont, X_cat], axis=1)
    pred = model.predict(X)
    label = lb.inverse_transform(pred)[0]

    return {"prediction": label, "predicted_by": current_user.username}
