FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8501

WORKDIR /app
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
EXPOSE 8501
CMD ["sh", "-c", "exec python -m streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
