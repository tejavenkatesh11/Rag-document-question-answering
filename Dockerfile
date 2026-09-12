# syntax=docker/dockerfile:1
FROM python:3.12-slim

WORKDIR /app

# Install CPU-only torch from the dedicated index first. The default PyPI
# wheel pulls the CUDA runtime, which adds gigabytes this application never
# uses. Doing it in its own layer also keeps it cached across code changes.
RUN pip install --no-cache-dir torch==2.14.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# torch is already installed above; the pin in requirements.txt refers to the
# +cpu build, so pip treats the requirement as satisfied.
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model into the image so the container does not download
# ~80MB on first question, and works without outbound Hugging Face access.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/sample_docs/ ./data/sample_docs/
COPY app.py pytest.ini ./

EXPOSE 8501

# GROQ_API_KEY is supplied at runtime, never baked into the image.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; \
    urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true"]
