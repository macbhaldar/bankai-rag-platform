# Launch the API and the Streamlit UI together.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -x ".venv/Scripts/python.exe" ] && [ ! -x ".venv/bin/python" ]; then
  echo "virtualenv not found — create it first:  py -3 -m venv .venv && .venv/Scripts/pip install -r requirements.txt"
  exit 1
fi
PY=".venv/Scripts/python.exe"; [ -x "$PY" ] || PY=".venv/bin/python"

echo "Starting API on http://127.0.0.1:8000 (docs at /docs)"
"$PY" scripts/run_api.py &
API_PID=$!
trap 'kill $API_PID 2>/dev/null || true' EXIT

echo "Starting UI on http://localhost:8501"
"$PY" -m streamlit run app/ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
