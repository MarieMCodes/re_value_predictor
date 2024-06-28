FROM python:3.8.12-slim

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
CMD uvicorn api:app --reload --host 0.0.0.0 --port $PORT

# deploy locally
# CMD uvicorn api:app --reload --host 0.0.0.0
