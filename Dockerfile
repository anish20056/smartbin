FROM python:3.10-slim

WORKDIR /app

COPY requirements-backend.txt .
RUN pip install -r requirements-backend.txt

COPY . .

EXPOSE 7860

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
