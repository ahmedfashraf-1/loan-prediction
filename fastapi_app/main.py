import os
import re
import pickle
from pathlib import Path
from datetime import datetime
from fastapi import FastAPI, Request, Form, status, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
import pandas as pd


# ============================================================
# CONFIG
# ============================================================
# ============================================================
# DATABASE AUTO-CONFIG (SkySQL Cloud + Local SQLite fallback)
# ============================================================

import urllib.parse

# Read env vars (Railway / production)
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_USER = os.getenv("DB_USER")
DB_PASS = os.getenv("DB_PASS")
DB_NAME = os.getenv("DB_NAME")

# Decide if we use Cloud or Local
USE_CLOUD_DB = all([
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASS,
    DB_NAME
])

if USE_CLOUD_DB:
    print("🌍 Using SkySQL Cloud Database")

    # Fix special characters in password

    DB_URL = (
        f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        "?charset=utf8mb4&ssl_verify_cert=false"
    )

else:
    print("💻 Using Local SQLite Database (No ENV found)")

    DB_URL = "sqlite:///./local.db"

SESSION_SECRET = "SUPER_SECRET_KEY"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "credit_model.pkl"


# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Credit Risk Portal (Auth + Prediction API)")

app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# ============================================================
# DATABASE SETUP
# ============================================================

Base = declarative_base()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
engine = create_engine(
    DB_URL,
    pool_pre_ping=True,
    pool_recycle=280
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)


# ============================================================
# AUTH HELPERS
# ============================================================

def hash_pass(p):
    return pwd_context.hash(p)

def verify_pass(p, hashed):
    return pwd_context.verify(p, hashed)

def get_user(email: str):
    db = SessionLocal()
    try:
        return db.query(User).filter(User.email == email).first()
    finally:
        db.close()

def create_user(username: str, email: str, password: str):
    db = SessionLocal()
    try:
        user = User(
            username=username,
            email=email,
            hashed_password=hash_pass(password)
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user
    except IntegrityError:
        db.rollback()
        raise ValueError("Email already exists")
    finally:
        db.close()


# ============================================================
# MODEL LOADING
# ============================================================

model = None

try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
        print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None


def validate_features(features, expected_len=None):
    if not isinstance(features, (list, tuple)):
        raise HTTPException(status_code=400, detail="Features must be a list")

    # OPTIONAL check
    if expected_len and len(features) != expected_len:
        raise HTTPException(status_code=400, detail=f"Expected {expected_len} features")

    return features


def predict_model(features):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server.")

    try:
        return float(model.predict([features])[0])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


# ============================================================
# AUTH ROUTES
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    user = request.session.get("user")
    return templates.TemplateResponse("home.html", {"request": request, "user": user})


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None, "success": None})


@app.post("/login", response_class=HTMLResponse)
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    user = get_user(email.strip())

    if not user or not verify_pass(password, user.hashed_password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password.", "success": None}
        )

    request.session["user"] = {"email": user.email, "username": user.username}
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request, "error": None})


@app.post("/register", response_class=HTMLResponse)
def register(
    request: Request, username: str = Form(...), email: str = Form(...),
    password: str = Form(...), password2: str = Form(...)
):
    # EMAIL VALIDATION
    email_pattern = r"^[a-zA-Z0-9._%+-]+@gmail\.com$"
    if not re.match(email_pattern, email.strip()):
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Invalid email format. Must be name@gmail.com"}
        )

    # PASSWORD RULES
    if len(password) < 6:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must be at least 6 characters long."})
    if not re.search(r"[A-Z]", password):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must contain at least one capital letter."})
    if not re.search(r"[0-9]", password):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must contain at least one number."})
    if not re.search(r"[@$!%*?&,.#^+=_-]", password):
        return templates.TemplateResponse("register.html", {"request": request, "error": "Password must contain at least one special character."})

    if password != password2:
        return templates.TemplateResponse("register.html", {"request": request, "error": "Passwords do not match."})

    # USER CREATION
    try:
        create_user(username.strip(), email.strip(), password)
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": None, "success": "Account created successfully. Please login."}
        )
    except ValueError as ve:
        return templates.TemplateResponse("register.html", {"request": request, "error": str(ve)})


@app.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse("/", status_code=status.HTTP_302_FOUND)


# ============================================================
# PREDICTION API
# ============================================================

@app.post("/predict")
def predict(data: dict):
    features = validate_features(data.get("features"))
    score = predict_model(features)
    return {"prediction": score}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        df = pd.read_csv(file.file)
    except:
        raise HTTPException(400, "Invalid CSV file")

    preds = []
    for _, row in df.iterrows():
        features = validate_features(row.values.tolist())
        preds.append(predict_model(features))

    return {"rows": len(preds), "predictions": preds}
# test database
@app.get("/debug/users")
def debug_users():
    db = SessionLocal()
    try:
        users = db.query(User).all()
        return {"users": [u.email for u in users]}
    finally:
        db.close()

