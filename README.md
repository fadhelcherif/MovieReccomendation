# Movie Recommendation System

## Description
A modern movie recommendation web application with a FastAPI backend and a Streamlit frontend. The backend serves ML-powered recommendations, while the frontend provides an interactive poster-based interface for users to select favorites and view suggestions.

## Architecture

```
+-------------------+        REST API        +-------------------+
|   Streamlit App   | <-------------------> |    FastAPI App    |
|  (frontend, 8501) |                      |  (backend, 8000)   |
+-------------------+                      +-------------------+
```
- **frontend/**: Streamlit UI, Dockerfile, requirements.txt
- **backend/**: FastAPI app, ML model, Dockerfile, requirements.txt
- **docker/**: docker-compose.yml, .dockerignore
- **k8s/**: (optional) Kubernetes manifests

## Advanced Git Workflow

- **Branches:**
  - `main`: Production-ready code
  - `dev`: Integration branch for features and fixes
  - `feat/<feature-name>`: New features
  - `bugfix/<bug-name>`: Bug fixes
  - `hotfix/<hotfix-name>`: Urgent production fixes

- **Workflow:**
  1. Create a feature branch: `git checkout -b feat/<feature-name>`
  2. Commit and push changes: `git push origin feat/<feature-name>`
  3. Open a Pull Request (PR) to `dev`
  4. After review, merge PR into `dev`
  5. When stable, merge `dev` into `main`
  6. For bugs: `bugfix/<bug-name>` → PR to `dev`
  7. For urgent prod fixes: `hotfix/<hotfix-name>` → PR to `main` and `dev`

## Local Development

### Prerequisites
- Docker & Docker Compose
- Python 3.12 (for local runs)

### Run with Docker Compose
```sh
cd docker
docker compose up --build
```
- Backend: http://localhost:8001
- Frontend: http://localhost:8501

### Run Manually
```sh
# Backend
cd backend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload

# Frontend
cd frontend
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Linting & Tests
```sh
# Lint
cd backend
flake8 .

# Tests
pytest test_main.py
```

## Contributing
- Follow the branch naming conventions
- Write clear commit messages
- Ensure all tests pass before PR
- Keep code style consistent (flake8)

---

**Contact:** Maintainer: Your Name <your.email@example.com>
