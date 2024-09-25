FROM python:3.10-slim-bookworm

WORKDIR /app/frontend
COPY . .


RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    cmake \
    build-essential \
    gcc


RUN pip install --upgrade pip && \
    pip install --ignore-installed -r requirements.txt


EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "main.py"]