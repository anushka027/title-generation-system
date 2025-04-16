@echo off
echo Starting Title Generation Server...
echo.

REM Check if the virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate
    echo Installing dependencies...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install transformers pandas scikit-learn flask flask-cors tqdm numpy sentencepiece
) else (
    call venv\Scripts\activate
)

echo Starting server...
python app.py