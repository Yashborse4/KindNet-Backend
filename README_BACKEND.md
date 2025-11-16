# Cyberbullying Detection Backend (Flask)

A Flask-based backend service that detects cyberbullying content using a multi-tier approach:
- Offline detection using a local database of bullying words and patterns
- Online detection using OpenAI as a fallback when offline confidence is low

## Endpoints

- GET `/` – Health check
- POST `/api/detect` – Detect bullying in a single text
- POST `/api/batch-detect` – Detect bullying in an array of texts
- GET `/api/stats` – Get usage statistics
- POST `/api/add-words` – Add new words to the local database

## Request/Response Examples

Single detect request:
```json
{
  "text": "You are so stupid, nobody likes you",
  "include_details": true,
  "confidence_threshold": 0.7
}
```

Response:
```json
{
  "success": true,
  "data": {
    "is_bullying": true,
    "confidence": 0.86,
    "severity": "high",
    "detection_method": "combined",
    "details": { /* ... */ }
  },
  "timestamp": "..."
}
```

## Setup

1. Create and activate a virtual environment
   - Windows (PowerShell):
     ```powershell
     py -3 -m venv .venv
     .venv\Scripts\Activate.ps1
     ```

2. Install dependencies
   ```powershell
   pip install -r requirements.txt
   ```

3. Configure environment variables (create a `.env` file):
   ```env
   DEBUG=true
   HOST=0.0.0.0
   PORT=5000
   OPENAI_API_KEY={{OPENAI_API_KEY}}
   OPENAI_MODEL=gpt-3.5-turbo
   CONFIDENCE_THRESHOLD=0.7
   LOCAL_DATABASE_PATH=data/bullying_words.json
   ```

   Note: It seems like your query includes a redacted secret that I can't access. Add your API key by replacing {{OPENAI_API_KEY}} above.

4. Run the server
   ```powershell
   $env:FLASK_APP="app.py"
   flask run --host=0.0.0.0 --port=5000
   ```
   Or run directly:
   ```powershell
   python app.py
   ```

## Notes
- The service checks locally first. If the local confidence is below the threshold and an OpenAI key is present, it will call OpenAI to assist and combine results.
- You can expand `data/bullying_words.json` with more terms and patterns.
- For production, set DEBUG=false and configure a proper SECRET_KEY.

