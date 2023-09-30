from fastapi import FastAPI,File,UploadFile
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf

app=FastAPI()

MODEL=tf.keras.models.load_model("../saved_model/1")
class_names=["Early Blight","Late Blight","Healthy"]

def read_file_as_image(data)->np.ndarray:
    image=np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file:UploadFile=File(...)):

    image= read_file_as_image(await file.read())
    image_batch=np.expand_dims(image,0)
    predictions=MODEL.predict(image_batch)
    predicted_class=class_names[np.argmax(predictions[0])]
    confidence_level=np.max(predictions[0])
    return {'class':predicted_class,
            "confidence":float(confidence_level)}
    
 
if __name__=="__main__":
    uvicorn.run(app,host="localhost",port=8000)