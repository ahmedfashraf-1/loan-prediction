import os
import pymysql
pymysql.install_as_MySQLdb()

from fastapi import FastAPI, Request, Form, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import IntegrityError
from passlib.context import CryptContext
from datetime import datetime
import re

# =============================
# DATABASE CONFIG  (MySQL)
# =============================

DB_URL = "mysql+pymysql://root:sheta123@localhost:3306/depi_auth?charset=utf8mb4"
print("🔍 USING DATABASE URL =>", DB_URL)

SESSION_SECRET = "SUPER_SECRET_KEY"

# =============================
# APP SETUP
# =============================

app = FastAPI(title="Auth - Credit Risk Portal")
app.add_middleware(SessionMiddleware, secret_key=SESSION_SECRET)

templates = Jinja2Templates(directory="templates")

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

# =============================
# DATABASE SETUP
# =============================

Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

engine = create_engine(DB_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(120), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(bind=engine)

# =============================
# Helper functions
# =============================

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

# =============================
# ROUTES
# =============================

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
def register(request: Request, username: str = Form(...), email: str = Form(...),
             password: str = Form(...), password2: str = Form(...)):

    # =============================
    # VALIDATION RULES ADDED HERE ONLY
    # =============================

    # Validate Email Format
    email_pattern = r"^[a-zA-Z0-9._%+-]+@gmail\.com$"
    if not re.match(email_pattern, email.strip()):
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Invalid email format. Must be name@gmail.com"}
        )

    # Validate Password Strength
    if len(password) < 6:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must be at least 6 characters long."}
        )

    if not re.search(r"[A-Z]", password):
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must contain at least one capital letter."}
        )

    if not re.search(r"[0-9]", password):
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must contain at least one number."}
        )

    if not re.search(r"[@$!%*?&,.#^+=_-]", password):
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Password must contain at least one special character."}
        )

    if password != password2:
        return templates.TemplateResponse(
            "register.html",
            {"request": request, "error": "Passwords do not match."}
        )

    # =============================
    # USER CREATION
    # =============================
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