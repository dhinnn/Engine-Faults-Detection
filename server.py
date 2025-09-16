import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Reuse your existing model logic
from app import ModelManager, Config

# Initialize models (loads independently as per your updated app.py)
config = Config()
model_manager = ModelManager()

BASE_DIR = Path(__file__).parent
TMP_DIR = BASE_DIR / "tmp"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

TMP_DIR.mkdir(exist_ok=True)
STATIC_DIR.mkdir(exist_ok=True)
TEMPLATES_DIR.mkdir(exist_ok=True)

app = FastAPI(title="Engine Fault Detection (No Gradio)")

# Mount static files (for spectrogram images)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Templates
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def save_upload_to_tmp(upload: UploadFile) -> Path:
    suffix = Path(upload.filename).suffix or ".wav"
    tmp_path = TMP_DIR / f"upload_{uuid.uuid4().hex}{suffix}"
    with tmp_path.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return tmp_path


def generate_spectrogram_copy(audio_path: str) -> Optional[str]:
    plot_path = model_manager.visualize_spectrogram(audio_path)
    if not plot_path or not os.path.exists(plot_path):
        return None
    # copy to static with unique name
    dest = STATIC_DIR / f"spectrogram_{uuid.uuid4().hex}.png"
    shutil.copy(plot_path, dest)
    return f"/static/{dest.name}"


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "results": None,
            "single_result": None,
            "spectrogram_url": None,
            "classes": ", ".join(config.CLASSES),
            "audio_params": {
                "sr": config.SAMPLE_RATE,
                "duration": config.DURATION,
                "n_mels": config.N_MELS,
            },
            "models": {
                "cnn": bool(model_manager.cnnlstm_model),
                "rf": bool(model_manager.rf_model),
            },
        },
    )


@app.post("/predict_both", response_class=HTMLResponse)
async def predict_both(request: Request, audio_file: UploadFile = File(...)):
    tmp_path = save_upload_to_tmp(audio_file)
    try:
        results_md = model_manager.predict_both_models(str(tmp_path))
        spectrogram_url = generate_spectrogram_copy(str(tmp_path))
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": results_md,
                "single_result": None,
                "spectrogram_url": spectrogram_url,
                "classes": ", ".join(config.CLASSES),
                "audio_params": {
                    "sr": config.SAMPLE_RATE,
                    "duration": config.DURATION,
                    "n_mels": config.N_MELS,
                },
                "models": {
                    "cnn": bool(model_manager.cnnlstm_model),
                    "rf": bool(model_manager.rf_model),
                },
            },
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


@app.post("/predict_single", response_class=HTMLResponse)
async def predict_single(
    request: Request,
    audio_file: UploadFile = File(...),
    model_type: str = Form(...),
):
    tmp_path = save_upload_to_tmp(audio_file)
    try:
        if model_type == "CNN-LSTM":
            label, conf = model_manager.predict_with_cnnlstm(str(tmp_path))
        else:
            label, conf = model_manager.predict_with_rf(str(tmp_path))

        single_result = f"Prediction: {label}\nConfidence: {conf:.3f}"
        spectrogram_url = generate_spectrogram_copy(str(tmp_path))

        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "results": None,
                "single_result": single_result,
                "spectrogram_url": spectrogram_url,
                "classes": ", ".join(config.CLASSES),
                "audio_params": {
                    "sr": config.SAMPLE_RATE,
                    "duration": config.DURATION,
                    "n_mels": config.N_MELS,
                },
                "models": {
                    "cnn": bool(model_manager.cnnlstm_model),
                    "rf": bool(model_manager.rf_model),
                },
            },
        )
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True)
