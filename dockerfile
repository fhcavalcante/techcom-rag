# Dockerfile para a API (Componente B)

FROM python:3.8-slim

WORKDIR /techcom-rag

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5001
CMD ["python", "app/main.py"]
