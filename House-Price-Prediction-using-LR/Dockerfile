FROM python:3.10.15-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip

COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY regmodel.pkl .
COPY scaling.pkl .

COPY . .

EXPOSE 5001

CMD ["python", "app.py"]
