FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir discord.py sentence-transformers
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
CMD ["python", "main.py"]
