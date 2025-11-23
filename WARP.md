# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository & Runtime Overview

- **Monorepo:** Root `package.json` orchestrates a React frontend (`Frontend/`) and a Flask backend (`Backend/`).
- **Frontend:** React 18 + TypeScript chat UI with Tailwind CSS (`Frontend/src`), focused on real-time chat and AI safety indicators.
- **Backend:** Python/Flask cyberbullying detection API (`Backend/`) with:
  - A **legacy, non-versioned API** in `Backend/app.py` exposing `/`, `/api/detect`, `/api/batch-detect`, `/api/stats`, `/api/add-words`.
  - A **modern, versioned API** under `Backend/app/` with an app factory (`create_app`), blueprints (`/api/v1/...`), Pydantic schemas, centralized configuration, structured logging, caching, rate limiting, and monitoring.
- **Ports (dev):** Frontend on `http://localhost:3000`, backend on `http://localhost:5000`.
- For a high-level feature/endpoint overview, prefer the root `README.md`, `Backend/API_GUIDE.md`, and `Backend/MODERNIZATION_SUMMARY.md`.

## Common Commands

### Root (Node + orchestration)

From the repo root:

- **Install all dependencies**
  - `npm install` (root dev tooling, e.g. `concurrently`)
  - `npm run setup` → runs `npm run install:frontend` and `npm run install:backend`
- **Start both services (legacy backend entry)**
  - `npm start` → runs backend via `python Backend/app.py` and frontend via `npm start` in `Frontend/`
  - `npm run start:dev` → same but with `flask --debug` for the backend
- **Frontend-only**
  - `npm run start:frontend`
  - `npm run build:frontend`
  - `npm run test:frontend`
- **Backend-only (via npm scripts)**
  - `npm run start:backend` → `cd Backend && python app.py`
  - `npm run start:backend:dev` → `cd Backend && python -m flask run --debug`
  - `npm run test:backend` → `cd Backend && python -m pytest`

### Backend (Python / Flask)

From `Backend/`:

- **Environment & dependencies** (Windows / PowerShell)
  - `py -3 -m venv .venv`
  - `.venv\Scripts\Activate.ps1`
  - `pip install -r requirements.txt`
- **Run modern API via app factory** (recommended for new work)
  - `python run.py` → uses `app.create_app()` and `/api/v1/...` endpoints
  - Environment selection via `FLASK_ENV` (`development`, `production`, etc.) and standard `HOST`, `PORT`, `DEBUG` env vars.
- **Run legacy API** (used by current frontend & integration tests)
  - `python app.py` → serves `/`, `/api/detect`, `/api/batch-detect`, `/api/stats`, `/api/add-words`
- **Backend tests (pytest with coverage)**
  - `python -m pytest` or `pytest` (uses `pytest.ini`, targets `tests/`)
  - With coverage (default via `pytest.ini`): `pytest` already runs with `--cov=app --cov-report=html:htmlcov --cov-report=term-missing --cov-fail-under=80`
  - **By marker:**
    - Unit tests: `pytest -m unit`
    - Integration tests: `pytest -m integration`
    - API tests: `pytest -m api`
    - Smoke tests: `pytest -m smoke`
    - Skip slow tests: `pytest -m "not slow"`
  - **Single test file / test case (examples):**
    - `pytest tests/api/test_detection_endpoints.py`
    - `pytest tests/api/test_detection_endpoints.py::TestDetectionEndpoints::test_detect_single_text`
- **Linting / type checking** (tools are installed via `requirements.txt`)
  - Format: `black app tests`
  - Imports: `isort app tests`
  - Lint: `flake8 app tests`
  - Types: `mypy app`

### Frontend (React + TypeScript)

From `Frontend/`:

- Install deps: `npm install`
- Start dev server: `npm start`
- Run tests (Jest via CRA): `npm test` (then use the interactive filter to run a single test by name or filename)
- Build production bundle: `npm run build`
- Lint (via root script): from repo root run `npm run lint:frontend`

### Integration & Monitoring

- **Node-based integration tests (exercise legacy `/api/...` endpoints):**
  - Start backend: `cd Backend && python app.py`
  - In another terminal at repo root: `node test-integration.js`
- **Docker (backend only, modern stack)**
  - From `Backend/`:
    - Dev stack: `docker-compose up --build`
    - Production image: `docker build -t cyberbullying-api .`
    - Run production container (example):
      - `docker run -e FLASK_ENV=production -e OPENAI_API_KEY=... -p 5000:5000 cyberbullying-api`

## Backend Architecture (Modern vs Legacy)

### Modern Flask Application (`Backend/app`, `run.py`)

- **Entry point:** `Backend/run.py`
  - Adds `Backend/` to `PYTHONPATH`, creates the Flask app via `app.create_app(environment)`.
  - Reads `FLASK_ENV`, `HOST`, `PORT`, `DEBUG` env vars and starts the server.
- **App factory:** `Backend/app/__init__.py`
  - Initializes a global `ConfigManager` instance (see below).
  - Configures Flask (secret key, debug/testing flags, JSON settings, content limits).
  - Binds OpenAI-related settings from environment (`OPENAI_API_KEY`, `OPENAI_MODEL`, token/temperature limits).
  - Applies detection defaults (`CONFIDENCE_THRESHOLD`, `LOCAL_DATABASE_PATH`).
  - Reads API rate limiting and caching settings from JSON config and exposes them on `app.config`.
  - Registers the API v1 blueprint (`api_v1_bp`) at `/api/v1`.
  - Adds a JSON health check at `/health` returning status, service name, version, and environment.
  - Wires in:
    - `setup_logging` (structured logging to console & rotating files),
    - global error handlers (`register_error_handlers`),
    - middleware (`setup_middleware`),
    - monitoring/metrics (`setup_monitoring`),
    - CORS configuration based on `config/settings.json` (`api.cors.*`).
- **Configuration layer:** `Backend/app/utils/config_manager.py` + `Backend/config/*.json`
  - Loads base settings from `config/settings.json`, overlays environment-specific files (e.g. `config/environments/development.json`, `production.json`).
  - Loads all user-facing messages from `config/messages.json`.
  - Environment variables override JSON via a mapping (e.g. `CONFIDENCE_THRESHOLD` → `detection.thresholds.confidence.default`).
  - Provides dot-notation getters (`get`, `get_threshold`, `is_feature_enabled`) and message lookups (`get_message`).
  - Used by logging, detection services, and response builders for consistent behavior across environments.
- **Models / validation:** `Backend/app/models/schemas.py`
  - Pydantic models define request/response contracts:
    - `DetectionRequest`, `BatchDetectionRequest`, `AddWordsRequest` (validation, size limits, enums for severity and analysis mode).
    - `DetectionResponse`, `BatchDetectionResponse`, `StatisticsResponse`, `HealthCheckResponse`, `ErrorResponse`, `SuccessResponse`.
  - Validation errors are converted into consistent JSON via `ResponseBuilder.validation_error`.
- **Services and utilities:**
  - `Backend/app/utils/logger.py`
    - Centralized logging configuration, JSON formatter for production and a human-readable formatter for development.
    - Hooks request/response logging via `before_request` / `after_request` (skipping `/health`).
    - Provides `LoggerMixin`, `get_logger`, and `@log_performance` decorator.
  - `Backend/app/utils/response_builder.py`
    - Central response constructor for success, error, validation, rate limit, 404/405, 500, 503, and paginated responses.
    - Uses `ConfigManager` message keys (e.g. `api.responses.success.detection_complete`) for localization/consistency.
  - `Backend/app/api/v1/*` (not all files listed above, see `Backend/app/api`)
    - Blueprints for detection, management, and monitoring endpoints outlined in `Backend/API_GUIDE.md`:
      - Detection: `/api/v1/detect/`, `/api/v1/detect/batch`, `/api/v1/detect/analyze`, `/api/v1/detect/validate`.
      - Management: `/api/v1/manage/words`, `/api/v1/monitor/stats`.
      - Monitoring/metrics: `/api/v1/monitor/metrics`, `/api/v1/monitor/health`.
    - Endpoints are built around the Pydantic schemas and `ResponseBuilder` utilities.
  - `Backend/app/services/*`
    - Service layer (e.g. detection, monitoring) encapsulating business logic and delegating to the lower-level detectors in the repo root (`intelligent_detector.py`, `multilingual_detector.py`, etc.).
    - Designed to be testable and composable, with async batch processing support as described in `API_GUIDE.md` and `MODERNIZATION_SUMMARY.md`.

### Detection Engine & Optimization Layer (shared)

- Core detection logic lives in modules like `Backend/intelligent_detector.py`, `advanced_detector.py`, `multilingual_detector.py`, and related helpers (`debug_detector.py`, `detector_adapter.py`, `database_manager.py`).
- `Backend/OPTIMIZATION_SUMMARY.md` documents the algorithmic improvements:
  - Aho–Corasick automaton for multi-pattern phrase detection with gap tolerance.
  - Adaptive gap limits and word-importance weighting for phrase boundaries.
  - Multiple caches (patterns, keywords, phrase index, normalized text) for sub‑millisecond detection on typical inputs.
- The modern API and the legacy API both rely on these optimized detectors; when changing algorithm behavior, update detectors and keep API contracts (Pydantic schemas + `Frontend/src/types/api.ts`) aligned.

### Legacy Flask Application (`Backend/app.py`)

- Standalone Flask app configured via `Config` (`Backend/config.py`) and the `IntelligentBullyingDetector`.
- Endpoints:
  - `GET /` → health check returning status, service name, timestamp.
  - `POST /api/detect` → single-text detection using `detector.detect_bullying_enhanced`.
  - `POST /api/batch-detect` → batch detection with simple loop and per-item error handling.
  - `GET /api/stats` → aggregate statistics from `detector.get_enhanced_statistics()`.
  - `POST /api/add-words` → validates and persists new bullying words via `detector.add_bullying_words`.
- Used by:
  - `Frontend/src/services/api.ts` (which currently targets `/api/detect`, `/api/batch-detect`, `/api/stats`, `/api/add-words`).
  - `test-integration.js` (Node integration tests against the legacy endpoints).
- For new backend feature work, prefer the modern app under `Backend/app/` and consider the legacy app as a compatibility layer and reference implementation.

### Testing & Observability (Backend)

- Pytest configuration in `Backend/pytest.ini`:
  - Discovers tests in `Backend/tests` with filenames `test_*.py` / `*_test.py`.
  - Enforces strict markers/config and 80% coverage via `--cov-fail-under=80`.
  - Defines markers: `unit`, `integration`, `api`, `slow`, `requires_openai`, `requires_redis`, `smoke`.
  - Sets compact tracebacks (`--tb=short`) and multiple coverage outputs (HTML, XML, terminal).
- Monitoring/metrics (modern API):
  - Prometheus-style metrics endpoints and a monitoring stack (Prometheus + Grafana) are wired via Docker Compose as described in `Backend/API_GUIDE.md` and `Backend/MODERNIZATION_SUMMARY.md`.

## Frontend Architecture (React Chat UI)

- **Entry & layout:**
  - `Frontend/src/index.tsx` mounts the app; `App.tsx` composes the chat layout.
  - Tailwind + custom CSS (`Frontend/src/index.css`, `App.css`) implement a glass-morphism theme and responsive layout.
- **Core components (`Frontend/src/components`)**
  - `ChatApp.tsx`
    - Maintains chat state (`messages`, typing/loading flags, detection enabled/disabled).
    - Uses the `api.quickDetect` helper to invoke the backend detection endpoint and derives safety feedback (flag, confidence).
    - Renders assistant messages summarizing detection results and warnings; toggling detection logs a system message.
  - `Message.tsx`
    - Presentation of a single chat message (user vs assistant) with timestamp and optional avatar.
  - `MessageInput.tsx`
    - Handles user input and submit; delegates to `handleSendMessage` in `ChatApp`.
  - `ChatHeader.tsx`
    - Header with the monitoring toggle (`detectionEnabled`) and status UI.
- **Typed API client (`Frontend/src/services/api.ts` + `Frontend/src/types/api.ts`)**
  - `CyberBullyingAPI` class wraps `fetch` with:
    - Base URL from `REACT_APP_API_URL` (defaults to `http://localhost:5000`).
    - Timeout + retry logic with exponential backoff.
    - Error normalization into an `ApiError` class.
  - Main methods:
    - `healthCheck()` → calls backend `/` health check (legacy endpoint).
    - `detectBullying()`, `batchDetectBullying()`, `getStats()`, `addBullyingWords()` → map directly to the legacy Flask `/api/...` endpoints.
    - `quickDetect(text, confidenceThreshold)` → thin helper used by `ChatApp` to call `detectBullying` and convert errors into a `{ result, error }` shape for UI use.
  - `types/api.ts` defines shared types for detection results, batch responses, statistics, and chat messages; keep these in sync with backend responses when changing the API.

## Configuration & Environments

### Backend

- **`.env` (in `Backend/`, from `.env.example`):**
  - `OPENAI_API_KEY` / `OPENAI_MODEL` → OpenAI integration (optional but required for full functionality).
  - `HOST`, `PORT`, `DEBUG` → server config.
  - `CONFIDENCE_THRESHOLD`, `LOCAL_DATABASE_PATH` → detection defaults.
  - `CORS_ORIGINS` → allowed origins for the frontend.
- **JSON config (`Backend/config`)**
  - `settings.json` → core application & detection settings (thresholds, feature flags, rate limits, caching).
  - `messages.json` → all user-facing/localized messages used by `ResponseBuilder`.
  - `environments/development.json`, `environments/production.json` → overrides.
  - Additional detection-related configs (e.g. `detection_features.json`, `language_detection.json`, `text_normalization.json`) parameterize aspects of the detection pipeline.

### Frontend

- **`.env` (in `Frontend/`, from `.env.example`):**
  - `REACT_APP_API_URL` → base URL used by `CyberBullyingAPI` (should match backend).
  - `REACT_APP_API_TIMEOUT` → request timeout in ms.
  - Feature flags such as `REACT_APP_ENABLE_VOICE_MESSAGES`, `REACT_APP_ENABLE_FILE_UPLOAD` (used for future UI features).

## How Pieces Fit Together

- In typical development:
  - Backend is started via `python Backend/app.py` (legacy) or `python Backend/run.py` (modern) on port 5000.
  - Frontend is started via `cd Frontend && npm start` on port 3000.
  - `Frontend/src/services/api.ts` calls the backend using `REACT_APP_API_URL` and `/api/...` routes.
  - Detection requests flow:
    - Frontend `ChatApp` → `api.quickDetect()` → backend `/api/detect` → detection engine (`IntelligentBullyingDetector` & related modules) → JSON response → `ChatApp` safety message.
- For new backend capabilities or API surface changes, implement them in the modern app (`Backend/app` + `/api/v1/...`) and then:
  - Add or adjust Pydantic schemas in `app/models/schemas.py`.
  - Expose new endpoints under the v1 blueprint.
  - Update the frontend `api.ts` client and `types/api.ts` to reflect response shapes (keeping the legacy `/api/...` endpoints stable while the frontend still uses them).
- For algorithmic changes, coordinate updates between:
  - Detector modules (`intelligent_detector.py`, etc.),
  - Optimization/config JSON under `Backend/config`, and
  - Any tests under `Backend/tests` and `test-integration.js` that encode expected behavior.
