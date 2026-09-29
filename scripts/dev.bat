@echo off
rem Launch the API (uvicorn, port 8000) and the Streamlit UI (port 8501) together.
cd /d "%~dp0.."

if not exist ".venv\Scripts\python.exe" (
  echo virtualenv not found - create it first:  py -3 -m venv .venv ^&^& .venv\Scripts\pip install -r requirements.txt
  exit /b 1)

echo Starting API on http://127.0.0.1:8000 (docs at /docs)
start "BankRAG API" .venv\Scripts\python.exe scripts\run_api.py

echo Starting UI on http://localhost:8501
.venv\Scripts\python.exe -m streamlit run app\ui\streamlit_app.py --server.port 8501 --server.address 0.0.0.0
