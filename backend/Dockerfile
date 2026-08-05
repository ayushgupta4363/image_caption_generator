FROM python:3.12-slim

WORKDIR /app

ENV TF_ENABLE_ONEDNN_OPTS=0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# RUN python -c "import tensorflow as tf; tf.keras.applications.VGG16(weights='imagenet')"

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]