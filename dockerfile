# ------------------------------
# Base image
# ------------------------------
FROM python:3.11-slim

# ------------------------------
# Environment
# ------------------------------
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# ------------------------------
# Install system dependencies
# ------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git curl \
    && rm -rf /var/lib/apt/lists/*

# ------------------------------
# Set working directory
# ------------------------------
WORKDIR /work

# ------------------------------
# Install Python dependencies
# Use BuildKit secret for private GitHub repos
# ------------------------------
# Make sure BuildKit is enabled:
# export DOCKER_BUILDKIT=1
# docker build --secret id=gh_token -t error-demo:latest .
RUN --mount=type=secret,id=gh_token \
    export GITHUB_TOKEN=$(cat /run/secrets/gh_token) && \
    python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
        "tab-err @ git+https://${GITHUB_TOKEN}@github.com/calgo-lab/tab_err.git@18c2c6d#egg=tab-err" \
        "conformal-data-cleaning @ git+https://${GITHUB_TOKEN}@github.com/calgo-lab/conformal-data-cleaning.git@d3d0008" \
        "mechdetect @ git+https://${GITHUB_TOKEN}@github.com/calgo-lab/MechDetect.git@d91ffc1" \
        streamlit itables plotly openpyxl numpy>=2.0.2 openml>=0.15.1 pandas>=2.3.3 quantile-forest>=1.4.1 scikit-learn>=1.6.1 typer-slim>=0.20.0

# ------------------------------
# Copy application code
# ------------------------------
COPY . .

# ------------------------------
# Expose Streamlit port
# ------------------------------
EXPOSE 8501

# ------------------------------
# Run Streamlit
# ------------------------------
CMD ["python","-m","streamlit","run","/work/app/Home.py","--server.port=8501","--server.address=0.0.0.0","--server.enableCORS=false"]

