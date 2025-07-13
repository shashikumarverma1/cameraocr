from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
from fastapi import FastAPI
from pydantic import BaseModel
import base64
import cv2
import numpy as np
from ultralytics import YOLO
import easyocr
# import cv2
# import numpy as np

app = FastAPI()

# Allow requests from React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Use specific IP in production
    allow_methods=["*"],
    allow_headers=["*"],
)
model =  YOLO("assets/anpr2_yolov9_int8.tflite")
ocr_reader = easyocr.Reader(['en'])

@app.get("/")
async def root():
    return [{"message": "API is working"} , {"message": "API is working"} , {"message": "API is working"}]

class ImageData(BaseModel):
    image:str

@app.post("/detect")
def submit_user(image: ImageData):
    import base64
    import numpy as np
    import cv2

    # Step 1: Decode the image
    image_data = base64.b64decode(image.image)
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Step 2: Run object detection
    results = model.predict(img_rgb, imgsz=640, conf=0.5, int8=True)

    plate_text = ""  # Initialize to empty string

    # Step 3: Process each detection
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            plate_crop = img_rgb[y1:y2, x1:x2]

            # Step 4: OCR on cropped plate
            ocr_result = ocr_reader.readtext(plate_crop)
            if ocr_result:
                plate_text =  " ".join([item[1] for item in ocr_result])
                print(ocr_result , "ocr_result")
                break  # Stop after first OCR match
        if plate_text:
            break  # Break outer loop if plate found

    return {"text": plate_text}


