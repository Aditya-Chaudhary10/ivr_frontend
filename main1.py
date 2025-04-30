from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from gtts import gTTS
from pydub import AudioSegment
from pydub.utils import which
import openai
import os
import io
import time

# Allow duplicate OpenMP libraries (for Whisper/pydub compatibility)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ========== Config ==========
GROQ_API_KEY = "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"
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi"}
TEMP_DIR = "tmp_audio"
os.makedirs(TEMP_DIR, exist_ok=True)

# Set the FFmpeg executable path for pydub
AudioSegment.converter = which("ffmpeg")

# ========== Init ==========
app = FastAPI()

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

model = WhisperModel("base", device="cpu", compute_type="int8")
client = openai.OpenAI(api_key=GROQ_API_KEY, base_url="https://api.groq.com/openai/v1")

# ========== API ==========
@app.post("/process-speech")
async def process_speech(
    audio_file: UploadFile = File(...),
    lang: str = Form("en"),
    speed: float = Form(1.0),
    volume: float = Form(1.0)
):
    # Step 1: Save audio
    audio_path = os.path.join(TEMP_DIR, f"input_{int(time.time())}.wav")
    with open(audio_path, "wb") as f:
        f.write(await audio_file.read())

    # Step 2: Transcribe using Whisper
    segments, _ = model.transcribe(audio_path, language=lang if lang != "en" else None)
    user_text = " ".join([seg.text for seg in segments]).strip()
    if not user_text:
        return {"error": "No speech detected."}

    # Step 3: LLM response via Groq
    system_prompt = {
        "role": "system",
        "content": f"You are a helpful voice assistant that responds in {SUPPORTED_LANGUAGES.get(lang, 'English')}."
    }
    messages = [
        system_prompt,
        {"role": "user", "content": user_text}
    ]
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        max_tokens=120,
        temperature=0.7
    )
    reply_text = completion.choices[0].message.content

    # Print the response in the terminal
    print("User said:", user_text)
    print("Assistant response:", reply_text)

    # Step 4: Generate TTS using gTTS
    chunks = split_text_for_tts(reply_text)
    combined_audio = None
    for chunk in chunks:
        tts = gTTS(text=chunk, lang=lang, tld="co.in" if lang == "hi" else "com", slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        segment = AudioSegment.from_file(fp, format="mp3")
        if combined_audio is None:
            combined_audio = segment
        else:
            combined_audio += segment

    # Adjust speed and volume
    if combined_audio:
        # More granular speed control
        combined_audio = combined_audio._spawn(combined_audio.raw_data, overrides={
            "frame_rate": int(combined_audio.frame_rate * speed)
        }).set_frame_rate(combined_audio.frame_rate)

        if volume != 1.0:
            combined_audio += int(10 * (volume - 1.0))

    # Step 5: Return audio and text
    out_fp = io.BytesIO()
    combined_audio.export(out_fp, format="mp3")
    out_fp.seek(0)
    
    # Return the audio stream
    return StreamingResponse(
        out_fp, 
        media_type="audio/mpeg",
        headers={
            "X-Transcription": user_text,
            "X-Response": reply_text
        }
    )

# ========== Helper ==========
def split_text_for_tts(text, max_chunk_size=150):
    sentences = text.replace("ред", ".").split(".")
    chunks, current = [], ""
    for sentence in sentences:
        if not sentence.strip():
            continue
        if len(current) + len(sentence) < max_chunk_size:
            current += sentence + "."
        else:
            chunks.append(current)
            current = sentence + "."
    if current:
        chunks.append(current)
    return chunks