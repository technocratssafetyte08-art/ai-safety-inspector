# AI Safety Inspector — Website Edition

This package is for a website only. It does not include Android APK packaging files.

## Contents
- `web/` — responsive website interface
- `backend/` — AI image-analysis API required by the website

## Important
The website needs the backend because the AI API key must stay on the server, not in browser JavaScript. This ZIP is source code; it is not yet hosted online.

## Run locally
1. Install Python 3.10 or newer.
2. Open a terminal in `backend/`.
3. Run: `pip install -r requirements.txt`
4. Set `OPENAI_API_KEY` as a server environment variable.
5. Run: `uvicorn main:app --host 0.0.0.0 --port 8000`
6. In another terminal, open `web/` and run: `python -m http.server 5500`
7. Visit `http://localhost:5500`.

The website currently points to `http://localhost:8000`. For public hosting, deploy the backend to a secure HTTPS server and change `API_BASE` in `web/app.js` to that backend URL.

Before production use, configure restricted CORS, authentication, upload limits, HTTPS, privacy/data-retention controls, and test on actual workplace images. AI output is only a preliminary visual screening and must be verified by a competent safety professional.
