FROM python:3.12-slim

WORKDIR /app

# 1. Install system dependencies for HDBSCAN
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Copy only the requirements file first (Best Practice)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Precision Copy: Only grab the 'app' folder
# This ignores .venv, .databricks, and all the extra root files
COPY ./app ./app

# 4. Set the PYTHONPATH so the container knows where 'app' is
ENV PYTHONPATH="${PYTHONPATH}:/app"

EXPOSE 8000

# 5. Run the API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]