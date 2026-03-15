# ---------------------------
# Base image
# ---------------------------
FROM python:3.12-slim

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

# ---------------------------
# Install Poetry
# ---------------------------
RUN python3 -m ensurepip && python3 -m pip install --upgrade pip \
    && pip install poetry==1.7.1

# ---------------------------
# Copy dependencies and install
# ---------------------------
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --only main

# ---------------------------
# Copy project code
# ---------------------------
COPY liver_cirrhosis_model ./liver_cirrhosis_model
COPY README.md ./README.md

# ---------------------------
# Create directories for data and models
# ---------------------------
RUN mkdir -p /app/data /app/model

# ---------------------------
# Entrypoint
# ---------------------------
ENTRYPOINT ["python", "liver_cirrhosis_model/model.py"]
CMD ["train", "--dataset", "/app/data/train.csv"]