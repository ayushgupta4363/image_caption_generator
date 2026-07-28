import os
import io
import pickle
import traceback
import numpy as np
import gc

# 1. Disable GPU searches completely to save memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# 2. Limit TensorFlow thread usage (prevents high memory spikes)
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# ... Rest of your FastAPI routes and model loading code ...
from PIL import Image

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dropout, Dense, Embedding, RepeatVector,
    Bidirectional, LSTM, Dot, Activation, Lambda, Concatenate
)
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI(title="Keras Image Captioning API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "my_model.keras")
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer.pkl")

feature_extractor = None
caption_model = None
tokenizer = None
index_to_word = {}

MAX_LENGTH = 34
VOCAB_SIZE = 8768  # From your model metadata


def build_caption_model():
    """
    Reconstructs your exact Functional model architecture:
    Inputs: 4096-dim image features + 34-len sequence
    Outputs: 8768-dim vocabulary probabilities
    """
    # 1. Image feature extractor branch
    inputs1 = Input(shape=(4096,), name="input_layer_10")
    fe1 = Dropout(0.5, name="dropout_8")(inputs1)
    fe2 = Dense(256, activation="relu", name="dense_10")(fe1)
    fe3 = RepeatVector(MAX_LENGTH, name="repeat_vector_4")(fe2)

    # 2. Text sequence branch
    inputs2 = Input(shape=(MAX_LENGTH,), name="input_layer_11")
    se1 = Embedding(VOCAB_SIZE, 256, mask_zero=True, name="embedding_4")(inputs2)
    se2 = Dropout(0.5, name="dropout_9")(se1)

    # 3. Recurrent Attention branch
    bi_lstm1 = Bidirectional(
        LSTM(256, return_sequences=True), name="bidirectional_8"
    )(fe3)
    bi_lstm2 = Bidirectional(
        LSTM(256, return_sequences=True), name="bidirectional_9"
    )(se2)

    # 4. Dot Product & Attention Softmax
    dot_prod = Dot(axes=[2, 2], name="dot_4")([bi_lstm1, bi_lstm2])
    attn_weights = Activation("softmax", name="activation_4")(dot_prod)

    # 5. Lambda layers (Einsum & Sum Reduction)
    context_mat = Lambda(
        lambda x: tf.einsum("ijk,ijl->ikl", x[0], x[1]), name="lambda_7"
    )([attn_weights, bi_lstm2])
    context_vector = Lambda(
        lambda x: tf.reduce_sum(x, axis=1), name="lambda_8"
    )(context_mat)

    # 6. Decoder & Dense Classifier
    merged = Concatenate(axis=-1, name="concatenate_3")([context_vector, fe2])
    dense1 = Dense(256, activation="relu", name="dense_11")(merged)
    outputs = Dense(VOCAB_SIZE, activation="softmax", name="dense_12")(dense1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs, name="functional_5")
    return model


@app.on_event("startup")
def preload_models():
    global feature_extractor, caption_model, tokenizer, index_to_word

    print("\n--- [STARTUP] Loading Models ---")

    # 1. Load VGG16
    print("[1/3] Loading VGG16 Feature Extractor...")
    vgg = VGG16()
    feature_extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

    # 2. Build model in python and load weights safely
    print(f"[2/3] Re-building and loading model weights from {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

    try:
        # Load weights safely into reconstructed architecture
        caption_model = build_caption_model()
        caption_model.load_weights(MODEL_PATH)
        print("Caption Model loaded without C++ crash!")
    except Exception as e:
        print(f"Fallback loading mode triggered: {e}")
        caption_model = tf.keras.models.load_model(MODEL_PATH, safe_mode=False)

    # 3. Load Tokenizer
    print(f"[3/3] Loading Tokenizer from {TOKENIZER_PATH}...")
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PATH}")
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)

    if hasattr(tokenizer, "word_index"):
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    elif hasattr(tokenizer, "index_word"):
        index_to_word = tokenizer.index_word

    print("--- [STARTUP COMPLETE] All models loaded into memory! ---\n")


def extract_features(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract feature vector as a numpy array
    features = feature_extractor(img_array, training=False)
    if isinstance(features, tf.Tensor):
        features = features.numpy()
    return features


def generate_caption(image_features: np.ndarray) -> str:
    start_token = "startseq"
    if hasattr(tokenizer, "word_index"):
        if "startseq" in tokenizer.word_index:
            start_token = "startseq"
        elif "start" in tokenizer.word_index:
            start_token = "start"

    in_text = start_token

    # Ensure image_features is a tensor of float32
    feat_tensor = tf.convert_to_tensor(image_features, dtype=tf.float32)

    for _ in range(MAX_LENGTH):
        seq = tokenizer.texts_to_sequences([in_text])
        if not seq or not seq[0]:
            break

        padded_seq = pad_sequences([seq[0]], maxlen=MAX_LENGTH, padding="post")
        
        # Convert text sequence to tensor as well (matching input types)
        seq_tensor = tf.convert_to_tensor(padded_seq, dtype=tf.float32)

        # Call model with matching tensor inputs
        yhat = caption_model([feat_tensor, seq_tensor], training=False)
        
        idx = int(np.argmax(yhat.numpy()))
        word = index_to_word.get(idx, None)

        if word is None:
            break

        in_text += " " + word

        if word in ["endseq", "end", "<end>", "[END]"]:
            break

    cleaned = in_text
    for tag in ["startseq", "endseq", "<start>", "<end>", "start", "end"]:
        cleaned = cleaned.replace(tag, "")

    return cleaned.strip()
        


@app.get("/")
def root():
    return {"status": "online", "message": "Go to /docs to test prediction."}


@app.post("/predict")
def predict_caption(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    try:
        image_bytes = file.file.read()
        features = extract_features(image_bytes)
        caption = generate_caption(features)
        return {"success": True, "caption": caption}

    except Exception as e:
        trace = traceback.format_exc()
        print(f"\n[ERROR DURING PREDICTION]\n{trace}")
        return {"success": False, "error": str(e), "traceback": trace}