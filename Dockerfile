FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install python-telegram-bot sentence-transformers deep-translator faiss-cpu requests
CMD ["python", "bot.py"]