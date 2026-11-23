FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
WORKDIR /app
COPY detector/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY common common
COPY detector .
CMD ["python", "main.py"]
