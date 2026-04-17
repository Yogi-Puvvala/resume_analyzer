# Resume Analyzer

A machine learning project that classifies resumes by job category. Upload a PDF resume and the app extracts the text, processes it, and returns a predicted job category — all served through a FastAPI backend and a Streamlit frontend, containerized with Docker and deployed on Render.

**Live App:** https://resume-analyzer-frontend-947v.onrender.com

---

## What This Project Does

Recruiters and job matching systems often need to quickly categorize resumes by domain — software engineering, data science, finance, marketing, and so on. This project automates that with a text classification model trained on resume data using TensorFlow and Keras.

What makes this project different from the previous ones in this portfolio is the input format. Rather than form fields or structured JSON, the user uploads a PDF file. The API handles the file upload, extracts the raw text using PyMuPDF, runs it through the same preprocessing pipeline used at training time, and returns the predicted category. The whole flow — from PDF upload to prediction — is handled end to end.

---

## Project Structure

```
resume_analyzer/
├── api/                      # FastAPI application (app.py)
├── frontend/                 # Streamlit app (streamlit_app.py)
├── models/                   # Trained model artifacts
├── src/                      # Text preprocessing logic
├── Resume_Analyzer.ipynb     # Training and exploration notebook
├── Dockerfile.api
├── Dockerfile.frontend
├── docker-compose.yaml
├── requirements_api.txt
└── requirements_frontend.txt
```

---

## How It Works

The user uploads a PDF resume through the Streamlit interface. The frontend sends the file to the FastAPI backend as a multipart form upload. The API extracts text from the PDF using PyMuPDF (`fitz`), applies the preprocessing pipeline from `src/`, and feeds the processed text into the trained TensorFlow/Keras model. The predicted job category is returned to the frontend and displayed to the user.

The preprocessing logic in `src/` is shared between training in the notebook and inference in the API, so the model always sees input in the same format it was trained on.

---

## Tech Stack

| Layer | Tool |
|---|---|
| Model | TensorFlow 2.19.0, Keras 3.13.2 |
| PDF extraction | PyMuPDF (fitz) |
| File upload handling | python-multipart |
| Preprocessing | Scikit-learn 1.8.0, NumPy |
| API | FastAPI, Uvicorn, Pydantic |
| Frontend | Streamlit |
| Containerization | Docker, Docker Compose |
| Deployment | Render |

The API and frontend have separate Dockerfiles and separate requirements files. PyMuPDF, TensorFlow, and all model dependencies are confined to the API container. The frontend image only installs Streamlit and requests, keeping it small.

---

## Running Locally

**Prerequisites:** Docker and Docker Compose installed.

```bash
git clone https://github.com/Yogi-Puvvala/resume_analyzer.git
cd resume_analyzer
docker-compose up --build
```

This starts two services:

- FastAPI backend at `http://localhost:8000`
- Streamlit frontend at `http://localhost:8501`

The frontend depends on the API passing its health check before it starts. The health check uses a 60-second start period with up to 5 retries to give TensorFlow enough time to load the model on startup.

---

## Training the Model

The model was built and trained in `Resume_Analyzer.ipynb`. Open it in Jupyter or Google Colab to explore the dataset, run preprocessing, train the model, and evaluate it. After training, save the model artifacts to the `models/` directory so the API can load them.

---

## Deployment

The API and frontend are deployed as separate services on Render, each with its own Dockerfile. The frontend is configured to point at the live API via the `API_URL` environment variable. Because Render's free tier spins down idle services, the first request after a period of inactivity may take a minute while the container restarts and TensorFlow reloads the model.

---

## Notes

- The API accepts PDF files only. Text extraction is handled by PyMuPDF on the server side, so the user does not need to copy and paste resume text manually.
- File uploads are handled through FastAPI's multipart form support, enabled by the `python-multipart` dependency.
- The `src/` preprocessing code runs identically at training time and inference time. Any change to the preprocessing logic needs to be reflected in both places and the model retrained.
