from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL= tf.keras.models.load_model("./models/first_model.h5")
CLASS_NAMES = {
    0: 'Actinic Keratosis',    # akiec
    1: 'Basal Cell Carcinoma', # bcc
    2: 'Benign Keratosis',     # bkl
    3: 'Dermatofibroma',       # df
    4: 'Melanoma',            # mel
    5: 'Melanocytic Nevi',    # nv
    6: 'Vascular Lesion'      # vasc
}

@app.get("/ping")    
async def ping():
    return "Alive!!"

def read_file_as_image(data) ->np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((128, 128))
    return image


@app.post("/predict")
async def predict(
    file: UploadFile =File(...)
):    
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image,0)
    img_batch = img_batch / 255.0
    predictions = MODEL.predict(img_batch)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    return {
        'class': predicted_class,
        'confidence': float(confidence)
    }

if __name__=='__main__':
    uvicorn.run(app,host='localhost',port=8081)