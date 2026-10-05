# Paper Screening and Data Extraction Pipeline

A configurable pipeline supporting two modules:

- **Screening:** evaluates titles and abstracts against eligibility criteria, with optional human review, a second LLM reviewer, and adjudication.
- **Data extraction:** processes included full-text papers and extracts structured information. This module still needs some work.

## Requirements

The pipeline requires either:

- installing Ollama and choose a small local model such as Qwen3 (depending on your hardware)
- an API key for an OpenAI-compatible provider such as Groq. (requires an account which is free) 

## Run

After installing required components and setting up keys and parameters you can either run:
- ```python.exe -m uvicorn api:app --reload``` and acces it via browser at http://127.0.0.1:8000
- ```docker compose up -d --build``` and acces it via browser at http://127.0.0.1:8000
