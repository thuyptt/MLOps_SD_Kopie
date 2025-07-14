FROM python:3.11-slim


RUN apt-get update && \
    apt-get install --no-install-recommends -y build-essential gcc git curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY mlops_project_2024/ mlops_project_2024/
COPY models/ models/

RUN pip install --no-cache-dir -r requirements.txt

# Expose the port
EXPOSE 8080

# Entry point for the FastAPI app
CMD ["uvicorn", "mlops_project_2024.predict_model:app", "--host", "0.0.0.0", "--port", "8080"]
