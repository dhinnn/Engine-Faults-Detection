# Engine Fault Detection (Gradio)

Minimal repository containing the app code to run a local Gradio UI for engine fault detection using two models (CNN-LSTM and Random Forest).

## Current classes
For now, we only include 4 classes — Normal, ExhaustLeak, Misfire, Rodknock — since these are easy to distinguish. The other 3 classes will be added soon.

## Contents
- `app.py` — Gradio app entrypoint
- `server.py` — Optional Flask server (not required for Gradio usage)
- `requirements.txt` — Python dependencies
- `.gitignore` — Excludes large model artifacts and local environment files

## Not included (large artifacts)
The following model files are not committed to keep the repo light:
- `best_cnnlstm_model.keras`
- `random_forest_model.joblib`
- `model_config.pkl`

Place these files in the project root if you want to run predictions.

### Where to get the model files
- Download from Google Drive:
  - Link: https://drive.google.com/drive/folders/1U6w25jJsWhjMmWFxqQas6H1_Uw2IjRdW?usp=sharing
- Or export the trained models from the training notebook/Colab and download the following files:
  - `best_cnnlstm_model.keras`
  - `random_forest_model.joblib`
  - `model_config.pkl`
- Put them in the repository root (same folder as `app.py`). They are intentionally excluded via `.gitignore`.

### What happens if models are missing?
- The app will still start and will clearly print which files are missing in the terminal.
- In the UI:
  - If the CNN-LSTM model is missing, predictions for that model will say "CNN-LSTM model not available".
  - If the Random Forest model is missing, predictions for that model will say "Random Forest model not available".
  - The "Model Info" tab shows availability with ✅/❌.

## Quick start

1. Create/activate a Python 3.11 virtual environment.
2. Install dependencies:
   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   python app.py
   ```
4. Open the printed local URL (e.g., http://127.0.0.1:7860), upload an engine audio file, and view predictions.

## TL;DR (simple run)
```bash
pip install -r requirements.txt
python app.py
```

## Notes
- For Apple Silicon, `tensorflow-macos` and `tensorflow-metal` are specified in `requirements.txt`.
- If `soundfile` complains about `libsndfile`, install via Homebrew:
  ```bash
  brew install libsndfile
  ```
