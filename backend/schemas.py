# app/schemas.py
from pydantic import BaseModel,Field

class UserCreate(BaseModel):
    name:str 
    phone_number:str 
    username: str
    password: str

class UserLogin(BaseModel):
    username: str
    password: str
    
    
