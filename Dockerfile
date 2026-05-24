FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install discord.py sentence-transformers torch
CMD ["python", "bot.py"]
