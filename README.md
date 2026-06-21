# AuraTriage भारत
![CI](https://github.com/sachanshivali-max/AuraTriage-Bharat/actions/workflows/ci.yml/badge.svg)
[![GitHub Repo](https://img.shields.io/badge/GitHub-AuraTriage-Bharat-blue?logo=github)](https://github.com/sachanshivali-max/AuraTriage-Bharat)
Repository URL: https://github.com/sachanshivali-max/AuraTriage-Bharat
Clone this repository using SSH:
```bash
git clone git@github.com:sachanshivali-max/AuraTriage-Bharat.git
```
![Architecture Diagram](file:///C:/Users/shiva/.gemini/antigravity-ide/brain/5db3a382-68c6-448c-b0d4-325ccc33b6d1/architecture_diagram_1782064889429.png)
![UI Mockup](file:///C:/Users/shiva/.gemini/antigravity-ide/brain/5db3a382-68c6-448c-b0d4-325ccc33b6d1/ui_placeholder_1782057141589_1782057157997.png)

## Project Overview

**AuraTriage भारत** is a healthcare triage assistant built with Python, FastAPI, and a modern web UI. It allows users to input patient symptoms and receive AI‑generated preliminary assessments and next‑step recommendations. The backend leverages a large language model (LLM) via the Google AI API.

## Features

- Simple web interface for symptom entry
- Real‑time streaming responses using Server‑Sent Events (SSE)
- Dockerized deployment for easy scaling
- Extensible architecture for adding more medical domains

## Project Structure

```
Cpstone Project/
├── web_ui/                      # Demo web interface
│   ├── server.py                # FastAPI + SSE streaming server
│   └── index.html               # Premium dark glassmorphism UI
├── requirements.txt             # Python dependencies
├── requirements_triage.txt      # Minimal dependencies for triage
├── .env                         # API key configuration
└── README.md
```

## Quick Start

1. **Install dependencies**
```bash
pip install -r requirements_triage.txt
```
2. **Set up environment variables** – create a `.env` file with your Google API key:
```text
GOOGLE_API_KEY=your_key_here
```
3. **Run with Web UI (full demo)**
```bash
python -m uvicorn web_ui.server:app --reload --port 8000
# Open http://localhost:3000
```

## ⚠️ Disclaimer

Healthcare advice from this system is AI‑generated and should be validated by a qualified medical professional before any clinical decision is made. Always follow local regulations and best practices for patient care.
