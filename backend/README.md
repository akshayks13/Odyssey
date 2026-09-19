# Odyssey backend

See the [root README](../README.md) for full setup, architecture, and demo notes.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn api.main:app --reload --port 8000
pytest -q
```
