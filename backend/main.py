from fastapi import FastAPI,UploadFile,File,Depends
import tensorflow as tf
from fastapi.responses import JSONResponse
from PIL import Image
from backend.tf_model import model
import cv2
import numpy as np
import joblib
import os
import io
from backend.security import get_current_user
from fastapi.middleware.cors import CORSMiddleware
from backend.auth import router as auth_router
from backend.database import engine
from sqlmodel import SQLModel
import shutil

app = FastAPI(title='Brain Tumor')

@app.on_event("startup")
def on_startup():
    
    if not os.path.exists("userspace"):
        os.mkdir("userspace")
    else:
        shutil.rmtree("userspace")
        os.mkdir("userspace")

        
       
    SQLModel.metadata.create_all(engine)

app.include_router(auth_router)

os.environ['TF_ENABLE_ONEDNN_OPTS']='0'

model.load_weights('backend/BrDX.keras')

label_encoder=joblib.load('backend/le.pkl')







@app.post('/detect')
async def detect_cancer(file:UploadFile=File(...),current_user =Depends(get_current_user) )->str:
    file_bytes = await file.read()
    image = np.array(Image.open(io.BytesIO(file_bytes)))
    image_resized = cv2.resize(image,(64,64))/255
    predicted = model.predict(np.array([image_resized]))
    predicted_class_index = np.argmax(predicted,axis=1)
    predicted_class = label_encoder.inverse_transform(predicted_class_index)
    predicted_prob = predicted[0][predicted_class_index]
    return JSONResponse({"ErrorCode":0,"Data":{"prediction":predicted_class[0].upper()},"Message":"success"})
        

        
            
