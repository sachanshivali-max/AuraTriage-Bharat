FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements_triage.txt .
RUN pip install --no-cache-dir -r requirements_triage.txt

# Copy the server, logic, and web_ui directory
COPY triage_server.py .
COPY run_triage.py .
COPY web_ui/triage.html ./web_ui/triage.html

# Expose port 8080 (default for Cloud Run)
EXPOSE 8080

# Command to run the application
CMD ["python", "triage_server.py"]
