# app/auth.py
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select
from backend.models import User
from backend.schemas import UserCreate, UserLogin
from backend.database import get_session
from backend.security import get_password_hash, verify_password ,get_current_user
from fastapi.responses import JSONResponse
router = APIRouter()

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(user: UserCreate, session: Session = Depends(get_session)):
    # Check if the username already exists
    if not user:
        return JSONResponse({"Message":"Error","Data":{},"ErrorCode":1})
    statement = select(User).where(User.username == user.username)
    existing_user = session.exec(statement).first()
    if existing_user:
        raise JSONResponse({"Message":"User already exists","Data":{},"ErrorCode":1},status_code=200)
    
    # Hash the password and generate a unique API key
    hashed_password = get_password_hash(user.password)
    api_key = str(uuid.uuid4())
    
    new_user = User(name=user.name,username=user.username, hashed_password=hashed_password, api_key=api_key,phone_number=user.phone_number)
    session.add(new_user)
    session.commit()
    session.refresh(new_user)
    
    return JSONResponse({"Message": "User registered successfully", "Data":{},"ErrorCode":0})

@router.post("/login")
def login(user: UserLogin, session: Session = Depends(get_session)):
    statement = select(User).where(User.username == user.username)

    db_user = session.exec(statement).first()
    
    
    if not db_user or not verify_password(user.password, db_user.hashed_password):
        return  JSONResponse({"message": "invalid credentials", "Data":{},"ErrorCode":1})
    else:
        return JSONResponse({"message": "Login successful", "Data":{"api_key":db_user.api_key},"ErrorCode":0})

@router.post("/profile")
def get_profile(current_user: User = Depends(get_current_user)):
    return JSONResponse({"Errorcode":0,"Data":current_user.model_dump(),"Message":"success"})
