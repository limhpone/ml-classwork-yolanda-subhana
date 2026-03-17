# 1. Use a lightweight Python base
FROM python:3.13-slim

# 2. Set the working directory
WORKDIR /app

# 3. Copy requirements and install them
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the model and app code
COPY model.pkl .
COPY app.py .
COPY templates/ ./templates/

# 5. Open port 5000
EXPOSE 5000

# 6. Run the app
CMD ["python", "app.py"]
