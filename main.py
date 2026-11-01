import os
import shutil
import tempfile
import traceback
import uuid
from pathlib import Path

import joblib
import whisper
from gtts import gTTS
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# --------------------------------------------------
# Paths and configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

RESPONSE_AUDIO_DIR = BASE_DIR / "response_audio"
RESPONSE_AUDIO_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "logistic_regression_model.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"


# --------------------------------------------------
# Emotion response messages
# --------------------------------------------------

EMOTION_RESPONSES = {
    "joy": "You sound happy! That is wonderful. Keep enjoying the moment!",
    "happy": "You sound happy! That is wonderful. Keep enjoying the moment!",
    "happiness": "You sound happy! That is wonderful. Keep enjoying the moment!",

    "sadness": "I am sorry you are feeling down. I hope things get better soon.",
    "sad": "I am sorry you are feeling down. I hope things get better soon.",

    "anger": "It sounds like something is frustrating you. Take a moment to breathe.",
    "angry": "It sounds like something is frustrating you. Take a moment to breathe.",

    "fear": "It sounds like you are feeling worried. You are not alone.",
    "fearful": "It sounds like you are feeling worried. You are not alone.",

    "neutral": "Thank you for sharing your audio. I hope you are having a good day.",
}

DEFAULT_RESPONSE = (
    "Thank you for sharing your audio. "
    "I hope you are having a good day."
)


# --------------------------------------------------
# FastAPI app
# --------------------------------------------------

app = FastAPI(title="VoxAI Speech Emotion Recognition API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
        "http://127.0.0.1:8001",
        "http://localhost:8001",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Serve generated response MP3 files
app.mount(
    "/response_audio",
    StaticFiles(directory=str(RESPONSE_AUDIO_DIR)),
    name="response_audio",
)


# --------------------------------------------------
# Load models
# --------------------------------------------------

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")

print("Loading emotion classification models...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

if not VECTORIZER_PATH.exists():
    raise FileNotFoundError(
        f"Vectorizer file not found: {VECTORIZER_PATH}"
    )

if not LABEL_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Label encoder file not found: {LABEL_ENCODER_PATH}"
    )

classifier = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

print("VoxAI backend is ready.")


# --------------------------------------------------
# Helper function
# --------------------------------------------------

def get_emotion_response(emotion: str) -> str:
    key = str(emotion).strip().lower()
    return EMOTION_RESPONSES.get(key, DEFAULT_RESPONSE)


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "VoxAI API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg")
    }


@app.post("/upload-audio")
async def upload_audio(file: UploadFile = File(...)):

    if not file:
        raise HTTPException(
            status_code=400,
            detail="No audio file was uploaded."
        )

    content_type = (file.content_type or "").lower()

    allowed_extensions = (
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac",
        ".webm",
        ".mp4",
    )

    filename = file.filename or "audio.wav"

    if (
        not content_type.startswith("audio/")
        and not filename.lower().endswith(allowed_extensions)
    ):
        raise HTTPException(
            status_code=400,
            detail="Please upload a valid audio file."
        )

    suffix = Path(filename).suffix or ".wav"
    temp_path = None

    try:

        # Read uploaded audio
        audio_bytes = await file.read()

        if not audio_bytes:
            raise HTTPException(
                status_code=400,
                detail="The uploaded audio file is empty."
            )

        # Save uploaded audio temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(audio_bytes)
            temp_path = temp_file.name

        # Check FFmpeg
        if shutil.which("ffmpeg") is None:
            raise RuntimeError(
                "FFmpeg was not found. "
                "Add its bin folder to PATH and restart the backend terminal."
            )

        # --------------------------------------------------
        # 1. Whisper transcription
        # --------------------------------------------------

        print(f"Transcribing: {filename}")

        transcription_result = whisper_model.transcribe(temp_path)

        transcript = (
            transcription_result.get("text") or ""
        ).strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail="Whisper could not detect speech in this audio."
            )

        print("Transcript:", transcript)

        # --------------------------------------------------
        # 2. Predict emotion
        # --------------------------------------------------

        text_features = vectorizer.transform([transcript])

        prediction = classifier.predict(text_features)[0]

        try:
            predicted_emotion = label_encoder.inverse_transform(
                [prediction]
            )[0]

        except (ValueError, TypeError, AttributeError):
            predicted_emotion = prediction

        predicted_emotion = str(predicted_emotion)

        print("Predicted emotion:", predicted_emotion)

        # --------------------------------------------------
        # 3. Generate response text
        # --------------------------------------------------

        response_text = get_emotion_response(
            predicted_emotion
        )

        print("Response text:", response_text)

        # --------------------------------------------------
        # 4. Generate AI voice using gTTS
        # --------------------------------------------------

        audio_filename = (
            f"response_{uuid.uuid4().hex}.mp3"
        )

        audio_path = RESPONSE_AUDIO_DIR / audio_filename

        tts = gTTS(
            text=response_text,
            lang="en"
        )

        tts.save(str(audio_path))

        print("Generated response audio:", audio_path)

        # --------------------------------------------------
        # 5. Return response to frontend
        # --------------------------------------------------

        return {
            "filename": filename,
            "transcript": transcript,
            "predicted_emotion": predicted_emotion,
            "response_text": response_text,
            "response_audio_url": (
                "http://127.0.0.1:8001/response_audio/"
                + audio_filename
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:

        print("Error while processing audio:")
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        ) from exc

    finally:

        # Delete temporary uploaded audio
        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except OSError:
                passtie