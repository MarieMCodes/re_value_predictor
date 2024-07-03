FROM python:3.10.14-bookworm

# Install system dependencies
RUN apt-get update && \
    apt-get install -y pkg-config libhdf5-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy all the needed files that we need to run the container
COPY requirements.txt requirements.txt
COPY project_code project_code
COPY models models
COPY setup.py setup.py


# What we want to run
RUN pip install --upgrade pip
RUN pip install -e .

# Change into the project_code directory to run uvicorn
WORKDIR "/project_code"

# deploy to GCP
# CMD uvicorn api:app --reload --host 0.0.0.0 --port $PORT

# deploy locally
CMD uvicorn api:app --reload --host 0.0.0.0
