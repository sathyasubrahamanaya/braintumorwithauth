# app/security.py
from fastapi import Header, Depends, HTTPException, status
from sqlmodel import Session, select
from backend.models import User
from backend.database import get_session
from passlib.context import CryptContext

# Password hashing setup
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

# Dependency to get the current user from the provided API key
def get_current_user(x_api_key: str = Header(...), session: Session = Depends(get_session)):
    statement = select(User).where(User.api_key == x_api_key)
    user = session.exec(statement).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, 
            detail="Invalid API Key"
        )
    return user
