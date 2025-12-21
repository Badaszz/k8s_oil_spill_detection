# main.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import onnxruntime as ort
import numpy as np
import requests
import cv2
from PIL import Image
import io
from pydantic import BaseModel

app = FastAPI()

# Load ONNX model
session = ort.InferenceSession(
    "model/unet_oilspill.onnx",
    providers=["CPUExecutionProvider"]
)

class ImageRequest(BaseModel):
    image_url: str

def preprocess_image(img: Image.Image, size=128):
    # Convert to grayscale
    img = img.convert("L")

    # Resize
    img = img.resize((size, size))

    # Convert to numpy
    img = np.array(img).astype(np.float32)

    # Normalize: match training (Normalize([0.5], [0.5]))
    img = (img / 255.0 - 0.5) / 0.5

    # Shape: (1,1,H,W)
    img = img[np.newaxis, np.newaxis, :, :]

    return img

def run_inference(img_tensor):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    preds = session.run(
        [output_name],
        {input_name: img_tensor}
    )[0]

    # Sigmoid + threshold
    preds = 1 / (1 + np.exp(-preds))
    mask = (preds > 0.5).astype(np.uint8)

    return mask[0, 0]  # (H,W)

@app.post("/predict")
def predict_from_url(req: ImageRequest):
    image_url = req.image_url

    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image URL")

    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    img_tensor = preprocess_image(img)
    mask = run_inference(img_tensor)

    # Create a blended (transparent) light-red overlay on top of the image
    base = np.array(img.resize((128, 128))).astype(np.float32)
    result = base.copy()

    # Light-red overlay color and alpha (0=transparent, 1=opaque)
    alpha = 0.6
    overlay_color = np.array([255, 30, 30], dtype=np.float32)

    mask_bool = (mask == 0)

    # Apply alpha blending only on mask regions so the red appears translucent
    result[mask_bool] = (
        (1 - alpha) * base[mask_bool] +
        alpha * overlay_color
    )

    overlay_img = Image.fromarray(result.astype("uint8"))

    buf = io.BytesIO()
    overlay_img.save(buf, format="PNG")
    buf.seek(0)

    return StreamingResponse(buf, media_type="image/png")


@app.get("/")
def root():
    return {"message": "Oil Spillage Detection Service"}


@app.get("/health")
def health():
    return {"status": "healthy asf"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)