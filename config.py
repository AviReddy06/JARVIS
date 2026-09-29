"""
Configuration settings for JARVIS Voice Assistant.
"""

# App Info
APP_NAME = "JARVIS - AI Voice Assistant"
APP_VERSION = "2.0.0"

# Audio Settings
SAMPLE_RATE = 16000              # Standard sample rate for sherpa-onnx & openWakeWord
CHANNELS = 1                     # Mono
CHUNK_SIZE = 1280                # Native block size for openWakeWord (80ms @ 16kHz)
DEFAULT_PAUSE_THRESHOLD = 0.9    # Seconds of silence after speech to trigger answer
MIN_SPEECH_DURATION = 0.45       # Minimum speech duration in seconds
BASE_ENERGY_THRESHOLD = 280      # Base RMS sensitivity threshold (optimized for laptop mics)
MAX_RECORDING_SECONDS = 40       # Safety limit

# Kokoro Voice Settings
DEFAULT_VOICE = "bm_george"      # Authentic British male voice (JARVIS)
# Other available voices: "bm_lewis", "am_adam", "am_echo", "am_michael", "af_bella", "af_nicole", "bf_emma"
DEFAULT_VOICE_SPEED = 1.0        # Authentic natural conversational speed
TTS_SAMPLE_RATE = 24000          # Kokoro native output sample rate
FOLLOW_UP_TIMEOUT_SECONDS = 10.0 # Seconds to stay awake waiting for follow-up questions without wake word

# openWakeWord Settings
WAKE_WORD_NAME = "hey_jarvis_v0.1.onnx"
DEFAULT_WAKEWORD_THRESHOLD = 0.12 # Optimized sensitivity for natural conversational speech
WAKE_WORD_REQUIRED = True        # Wait for "Hey Jarvis" before listening (set False for continuous live mode)
DEFAULT_INPUT_DEVICE = None      # None = system default; set to device index (e.g., 2 for Realtek) if needed

# Speech-to-Text Options (sherpa-onnx Streaming Zipformer)
DEFAULT_STREAMING_MODEL = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
STREAMING_NUM_THREADS = 2
STREAMING_RULE1_SILENCE = 2.4   # Trailing silence if no clear endpoint
STREAMING_RULE2_SILENCE = 0.85  # Trailing silence after speech to trigger instant LLM handoff
STREAMING_RULE3_MAX_LEN = 30.0  # Max duration of continuous speech segment
STREAMING_ENDPOINT_RULE1_SILENCE = STREAMING_RULE1_SILENCE
STREAMING_ENDPOINT_RULE2_SILENCE = STREAMING_RULE2_SILENCE
STREAMING_ENDPOINT_RULE3_MAX_LEN = STREAMING_RULE3_MAX_LEN
USE_STREAMING_ASR = True

# Ollama Settings
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_VISION_MODEL = "moondream"
DEFAULT_NUM_CTX = 8192               # Context window in tokens (default 8192, up to 16384/32768)
DEFAULT_MAX_CHAT_HISTORY = 30        # Messages preserved in active conversational memory

# Default System Prompt - Tuned for human-like spoken dialogue
DEFAULT_SYSTEM_PROMPT = (
    "You are Jarvis, a brilliant, articulate, and respectful AI assistant. "
    "Respond in clean, natural spoken English as if talking directly out loud in person. "
    "Speak in complete, natural sentences with proper punctuation. "
    "Keep answers concise (1 to 3 sentences unless asked for details). "
    "Do NOT use markdown headers, asterisks, bullet points, emoji, or code blocks in spoken responses."
)
