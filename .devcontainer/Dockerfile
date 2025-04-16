FROM mcr.microsoft.com/devcontainers/python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    NLTK_DATA=/workspaces/book-de/nltk_data \
    HF_HOME=/workspaces/book-de/hf_home

WORKDIR /workspaces/book-de

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    sqlite3 \
    libgl1 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN mkdir -p $NLTK_DATA $HF_HOME && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt 
