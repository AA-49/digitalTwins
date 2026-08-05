# Reproducible CPU environment for Jupyter training, the dashboard, and SMPL export.
FROM python:3.11-slim

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install CPU-only PyTorch from the official index before the project requirements.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.2.2+cpu

# Install dependencies before project code so source-only edits retain Docker's cache.
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# The licensed basicModel_*.pkl files are legacy SMPL assets serialized with
# Chumpy. Chumpy's legacy build metadata requires installation without pip's
# isolated builder on modern Python.
RUN pip install --no-build-isolation chumpy==0.70

# Copy project files after dependencies.
COPY . /app

# Create non-root user
RUN useradd -m appuser || true
USER appuser

ENV PATH="/home/appuser/.local/bin:${PATH}"

EXPOSE 5000 8888

CMD ["python", "app.py"]
