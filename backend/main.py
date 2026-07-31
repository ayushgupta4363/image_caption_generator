import os
import io
import pickle
import traceback
import numpy as np
import warnings
from dotenv import load_dotenv

# 1. Load environment variables from the .env file
load_dotenv()

# Suppress non-critical Keras/TF backend warnings
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Disable GPU searches completely to save memory
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

# Limit TensorFlow thread usage to avoid memory spikes
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub import hf_hub_download

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

# Hugging Face Configuration
HF_MODEL_ID = os.getenv("HF_MODEL_ID")
HF_API_TOKEN = os.getenv("HF_API_TOKEN")

feature_extractor = None
caption_model = None
tokenizer = None
index_to_word = {}

MAX_LENGTH = 34
VOCAB_SIZE = 8768


def build_caption_model():
    """
    Reconstructs your exact Functional model architecture:
    Inputs: 4096-dim image features + 34-len sequence
    Outputs: 8768-dim vocabulary probabilities
    """
    # 1. Image feature extractor branch
    inputs1 = Input(shape=(4096,), name="input_layer_10", dtype=tf.float32)
    fe1 = Dropout(0.5, name="dropout_8")(inputs1)
    fe2 = Dense(256, activation="relu", name="dense_10")(fe1)
    fe3 = RepeatVector(MAX_LENGTH, name="repeat_vector_4")(fe2)

    # 2. Text sequence branch
    inputs2 = Input(shape=(MAX_LENGTH,), name="input_layer_11", dtype=tf.float32)
    se1 = Embedding(VOCAB_SIZE, 256, mask_zero=False, name="embedding_4")(inputs2)
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
        lambda x: tf.einsum("ijk,ijl->ikl", x[0], tf.identity(x[1])), name="lambda_7"
    )([attn_weights, bi_lstm2])
    
    context_vector = Lambda(
        lambda x: tf.reduce_sum(tf.identity(x), axis=1), name="lambda_8"
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

    if not HF_MODEL_ID or not HF_API_TOKEN:
        print("[WARNING] HF_MODEL_ID or HF_API_TOKEN not found in .env file!")

    # 1. Load VGG16
    print("[1/3] Loading VGG16 Feature Extractor...")
    vgg = VGG16()
    feature_extractor = Model(inputs=vgg.inputs, outputs=vgg.layers[-2].output)

    # 2. Download and load model weights from Hugging Face Hub (or fallback locally)
    print(f"[2/3] Downloading model weights from HF Hub ({HF_MODEL_ID})...")
    try:
        model_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="my_model.keras",
            token=HF_API_TOKEN
        )
        caption_model = build_caption_model()
        caption_model.load_weights(model_path)
        print("Caption Model loaded successfully from Hugging Face!")
    except Exception as e:
        print(f"[FALLBACK] Failed to load from HF Hub ({e}). Looking for local file...")
        local_model_path = os.path.join(os.path.dirname(__file__), "my_model.keras")
        caption_model = build_caption_model()
        caption_model.load_weights(local_model_path)

    # 3. Download and load Tokenizer from Hugging Face Hub
    print(f"[3/3] Downloading Tokenizer from HF Hub ({HF_MODEL_ID})...")
    try:
        tokenizer_path = hf_hub_download(
            repo_id=HF_MODEL_ID,
            filename="tokenizer.pkl",
            token=HF_API_TOKEN
        )
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)
    except Exception as e:
        print(f"[FALLBACK] Failed to load tokenizer from HF Hub ({e}). Looking for local file...")
        local_tokenizer_path = os.path.join(os.path.dirname(__file__), "tokenizer.pkl")
        with open(local_tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

    if hasattr(tokenizer, "word_index"):
        index_to_word = {idx: word for word, idx in tokenizer.word_index.items()}
    elif hasattr(tokenizer, "index_word"):
        index_to_word = tokenizer.index_word

    print("--- [STARTUP COMPLETE] Ready for predictions! ---\n")


def extract_features(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    features = feature_extractor(img_array, training=False)
    if isinstance(features, tf.Tensor):
        features = features.numpy()
    return features


def generate_caption_advanced(
    image_features: np.ndarray, 
    beam_width: int = 3, 
    repetition_penalty: float = 1.25
) -> str:
    start_token = "startseq"
    if hasattr(tokenizer, "word_index"):
        if "startseq" in tokenizer.word_index:
            start_token = "startseq"
        elif "start" in tokenizer.word_index:
            start_token = "start"

    feat_tensor = tf.cast(tf.convert_to_tensor(image_features), dtype=tf.float32)
    if len(feat_tensor.shape) == 1:
        feat_tensor = tf.expand_dims(feat_tensor, axis=0)

    start_idx = tokenizer.texts_to_sequences([start_token])[0][0]
    beams = [([start_idx], 0.0)]
    
    stop_indices = {
        tokenizer.word_index[w]
        for w in ["endseq", "end", "<end>", "[END]"]
        if hasattr(tokenizer, "word_index") and w in tokenizer.word_index
    }

    for _ in range(MAX_LENGTH - 1):
        candidates = []
        for seq, score in beams:
            if seq[-1] in stop_indices:
                candidates.append((seq, score))
                continue

            padded_seq = pad_sequences([seq], maxlen=MAX_LENGTH, padding="pre")
            seq_tensor = tf.cast(tf.convert_to_tensor(padded_seq), dtype=tf.float32)

            preds = caption_model([feat_tensor, seq_tensor], training=False).numpy()[0]
            
            for token_id in set(seq):
                if preds[token_id] > 0:
                    preds[token_id] /= repetition_penalty

            last_token = seq[-1]
            if last_token in range(len(preds)):
                preds[last_token] /= 10.0

            preds_sum = np.sum(preds)
            if preds_sum > 0:
                preds = preds / preds_sum

            top_k_indices = np.argsort(preds)[-beam_width:]

            for idx in top_k_indices:
                prob = np.maximum(preds[idx], 1e-10)
                candidates.append((seq + [idx], score + np.log(prob)))

        ordered = sorted(
            candidates, 
            key=lambda x: x[1] / (len(x[0]) ** 0.7), 
            reverse=True
        )
        beams = ordered[:beam_width]

        if all(b[0][-1] in stop_indices for b in beams):
            break

    best_seq = beams[0][0]

    ignore_tokens = {"startseq", "endseq", "<start>", "<end>", "start", "end", "<pad>"}
    words = []
    for idx in best_seq:
        w = index_to_word.get(idx, "")
        if w and w.lower() not in ignore_tokens:
            words.append(w)

    return " ".join(words).strip()


def generate_caption(image_features: np.ndarray) -> str:
    return generate_caption_advanced(image_features, beam_width=3, repetition_penalty=1.25)


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