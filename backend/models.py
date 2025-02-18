# app/models.py
from sqlmodel import SQLModel, Field

class User(SQLModel, table=True):
    id: int = Field(default=None, primary_key=True)
    phone_number:str = Field(index=True)
    username: str = Field(index=True, unique=True)
    name:str
    hashed_password: str
    api_key: str = Field(unique=True, index=True)
