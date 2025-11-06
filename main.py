from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io, os

app = FastAPI(title="YOLOv5x Detection API")

# Cargar el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='models/best_yolov5_v5x.pt', force_reload=False)
model.conf = 0.25  # umbral de confianza

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Leer imagen subida
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Realizar inferencia
        results = model(image, size=640)

        # Convertir resultados a JSON
        detections = results.pandas().xyxy[0].to_dict(orient="records")

        # Guardar imagen con las detecciones
        os.makedirs("outputs/detections", exist_ok=True)
        save_path = f"outputs/detections/{file.filename}"
        results.save(save_dir="outputs/detections")

        return JSONResponse({
            "filename": file.filename,
            "detections": detections,
            "saved_image": save_path
        })

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
