"""
JARVIS Voice Assistant - Background HUD Service (PyQt6)
========================================================
Architectural Highlights:
- Background QThread (AudioListenerThread) handling sounddevice continuous stream,
  openWakeWord, sherpa-onnx streaming Zipformer ASR, Ollama LLM, and Kokoro TTS.
- Invisible QMainWindow HUD on startup with FramelessWindowHint, WindowStaysOnTopHint,
  Tool (no taskbar item), and WA_TranslucentBackground.
- Neon green glowing HUD QLabel dynamically centered at the bottom of the primary monitor.
- Real-time word-by-word streaming partial transcription token emissions to HUD.
- Signal-driven state transitions (wake_signal, update_text_signal, partial_text_signal, sleep_signal).
- QTimer-driven 4-second readability pause before smoothly fading out and hiding.
- System tray icon with test controls and graceful shutdown.
"""

import collections
import datetime
import math
import os
import struct
import sys
import time
from typing import Optional

# Safe stdout/stderr redirection and exception logging for pythonw.exe execution on Windows
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_hud.log")

class TeeLogger:
    def __init__(self, filename, stream):
        self.file = open(filename, "a", encoding="utf-8", buffering=1)
        self.stream = stream

    def write(self, message):
        if self.stream is not None:
            try:
                self.stream.write(message)
                self.stream.flush()
            except Exception:
                pass
        try:
            self.file.write(message)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        if self.stream is not None:
            try:
                self.stream.flush()
            except Exception:
                pass
        try:
            self.file.flush()
        except Exception:
            pass

sys.stdout = TeeLogger(LOG_FILE, sys.stdout)
sys.stderr = TeeLogger(LOG_FILE, sys.stderr)

def handle_exception(exc_type, exc_value, exc_traceback):
    import traceback
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
    except Exception:
        pass

sys.excepthook = handle_exception

import numpy as np
import sounddevice as sd
import threading
from PyQt6.QtCore import QBuffer, QIODevice, QEasingCurve, QObject, QPropertyAnimation, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QBrush, QColor, QFont, QIcon, QPainter, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QGraphicsDropShadowEffect,
    QLabel,
    QMainWindow,
    QMenu,
    QSystemTrayIcon,
)

import ollama

from config import (
    APP_NAME,
    BASE_ENERGY_THRESHOLD,
    CHANNELS,
    CHUNK_SIZE,
    DEFAULT_INPUT_DEVICE,
    DEFAULT_MAX_CHAT_HISTORY,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PAUSE_THRESHOLD,
    DEFAULT_STREAMING_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VISION_MODEL,
    DEFAULT_VOICE,
    DEFAULT_VOICE_SPEED,
    DEFAULT_WAKEWORD_THRESHOLD,
    FOLLOW_UP_TIMEOUT_SECONDS,
    MAX_RECORDING_SECONDS,
    MIN_SPEECH_DURATION,
    SAMPLE_RATE,
    STREAMING_ENDPOINT_RULE1_SILENCE,
    STREAMING_ENDPOINT_RULE2_SILENCE,
    STREAMING_ENDPOINT_RULE3_MAX_LEN,
    STREAMING_NUM_THREADS,
    USE_STREAMING_ASR,
    WAKE_WORD_NAME,
)
from ollama_engine import OllamaEngine
from streaming_asr import StreamingASREngine, normalize_asr_text
from tts_engine import TTSEngine
from wakeword_engine import WakeWordEngine


# =============================================================================
# Background Audio & AI Worker Thread (QThread)
# =============================================================================
class AudioListenerThread(QThread):
    """
    Background worker that monitors microphone audio, detects the wake word
    ('Hey Jarvis') using openWakeWord, runs real-time streaming Zipformer ASR
    with word-by-word UI updates, reasons with Ollama, and synthesizes voice with Kokoro TTS.
    
    Communicates with the GUI strictly via PyQt signals.
    """
    # Custom PyQt signals required by architecture specification
    wake_signal = pyqtSignal()
    update_text_signal = pyqtSignal(str)
    sleep_signal = pyqtSignal()
    partial_text_signal = pyqtSignal(str)  # Real-time word-by-word token emissions

    def __init__(
        self,
        streaming_model_name: str = DEFAULT_STREAMING_MODEL,
        ollama_model_name: str = DEFAULT_OLLAMA_MODEL,
        wake_word_required: bool = True,
        parent=None,
        **kwargs,
    ):
        super().__init__(parent)
        self.streaming_model_name = streaming_model_name
        self.ollama_model_name = ollama_model_name
        self.wake_word_required = wake_word_required

        self.is_running = True
        self.is_paused = False
        self.chat_history = []

        self.vision_model_name = DEFAULT_VISION_MODEL
        self.vision_available = False

        self.stream: Optional[sd.InputStream] = None
        self.wakeword_engine: Optional[WakeWordEngine] = None
        self.tts_engine: Optional[TTSEngine] = None
        self.ollama_engine = OllamaEngine(num_ctx=DEFAULT_NUM_CTX)
        self.interrupted = False

        # Real-time Streaming ASR Engine (Zipformer transducer via sherpa-onnx)
        self.use_streaming_asr = USE_STREAMING_ASR
        self.streaming_asr = StreamingASREngine(
            model_name=streaming_model_name,
            num_threads=STREAMING_NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            rule1_silence=STREAMING_ENDPOINT_RULE1_SILENCE,
            rule2_silence=STREAMING_ENDPOINT_RULE2_SILENCE,
            rule3_max_utterance=STREAMING_ENDPOINT_RULE3_MAX_LEN,
        )

    def trigger_interrupt(self):
        """Immediately stop speech and LLM generation upon user interruption."""
        self.interrupted = True
        if self.tts_engine:
            self.tts_engine.stop()
        self.ollama_engine.stop_generation()

    def run(self):
        """Thread execution entry point."""
        # 1. Initialize Real-Time Streaming ASR (Zipformer via sherpa-onnx)
        try:
            self.streaming_asr.load()
            print(f"[JARVIS ASR] Real-time Streaming Zipformer ASR Online: {self.streaming_model_name}", flush=True)
        except Exception as e:
            self.update_text_signal.emit(f"Streaming ASR model error: {e}")
            print(f"[JARVIS ASR Error] Streaming ASR load failed: {e}", flush=True)
            return

        # 2. Initialize openWakeWord
        try:
            self.wakeword_engine = WakeWordEngine(
                model_name=WAKE_WORD_NAME,
                threshold=DEFAULT_WAKEWORD_THRESHOLD,
            )
            self.wakeword_engine.load()
        except Exception as e:
            self.update_text_signal.emit(f"Wake word model error: {e}")

        # 3. Initialize Kokoro TTS
        try:
            self.tts_engine = TTSEngine(
                voice=DEFAULT_VOICE,
                speed=DEFAULT_VOICE_SPEED,
            )
            self.tts_engine.load()
        except Exception as e:
            self.update_text_signal.emit(f"TTS engine error: {e}")

        # 4. Resolve pulled Ollama models (LLM and Multimodal Vision)
        try:
            tags = ollama.list()
            available = [m.get("model", "") for m in tags.get("models", []) if m.get("model")]
            if available:
                matched = next((m for m in available if "llama3.2" in m.lower()), available[0])
                self.ollama_model_name = matched
                matched_vision = next((m for m in available if "moondream" in m.lower() or "vision" in m.lower() or "llava" in m.lower()), None)
                if matched_vision:
                    self.vision_model_name = matched_vision
                    self.vision_available = True
                    print(f"[JARVIS Vision] Screen Vision model ready: {self.vision_model_name}", flush=True)
        except Exception:
            pass

        # 5. Open sounddevice Input Stream
        try:
            device_index = DEFAULT_INPUT_DEVICE
            self.stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=CHUNK_SIZE,
                device=device_index,
            )
            self.stream.start()
            dev_name = "Default System Microphone"
            try:
                if device_index is not None:
                    dev_info = sd.query_devices(device_index)
                    dev_name = dev_info.get("name", dev_name)
                else:
                    dev_info = sd.query_devices(kind="input")
                    dev_name = dev_info.get("name", dev_name)
            except Exception:
                pass
            print(f"[JARVIS] Audio input stream opened via sounddevice: {dev_name} (index: {device_index})", flush=True)
        except Exception as e:
            self.update_text_signal.emit(f"Microphone initialization error: {e}")
            return

        # 6. Main Background Processing Loop
        zero_audio_counter = 0
        while self.is_running:
            try:
                # Flush residual audio buffer
                self._flush_stream()

                # --- Phase A: Wake Word Standby ---
                if self.wake_word_required:
                    wake_detected = False
                    while self.is_running and not wake_detected:
                        if self.is_paused:
                            time.sleep(0.04)
                            continue

                        try:
                            samples, _ = self.stream.read(CHUNK_SIZE)
                        except Exception:
                            continue

                        audio_slice = samples.flatten()
                        rms = float(np.sqrt(np.mean(audio_slice ** 2))) * 32767.0
                        if rms < 1.0:
                            zero_audio_counter += 1
                            if zero_audio_counter == 100:  # ~8 seconds of pure digital silence
                                print("[JARVIS WARNING] Microphone audio level is 0.0 (silence). Check if mic is hardware muted or switched off!", flush=True)
                        else:
                            zero_audio_counter = 0

                        pcm_bytes = (audio_slice * 32767.0).astype(np.int16).tobytes()
                        if self.wakeword_engine and self.wakeword_engine.process_pcm_chunk(pcm_bytes):
                            last_score = self.wakeword_engine.get_last_score()
                            print(f"[JARVIS] Wake word triggered! Score: {last_score:.3f}", flush=True)
                            wake_detected = True
                            break

                    if not self.is_running:
                        break

                # --- Phase B: Wake Word Triggered -> Active Conversation Session ---
                self._flush_stream()
                self.wake_signal.emit()
                self.update_text_signal.emit(
                    "⚡ <b>JARVIS Activated</b><br>"
                    "<span style='color:#00FFA3;font-size:13px;'>Yes sir? Standing by for your command...</span>"
                )

                # Vocal acknowledgment
                if self.tts_engine:
                    self.tts_engine.speak("Yes sir?", blocking=True)

                in_conversation = True
                is_first_turn = True
                last_reply = ""

                while self.is_running and in_conversation:
                    self._flush_stream()
                    prompt_header = (
                        "🎙️ <b>Listening...</b><br>"
                        if is_first_turn
                        else f"🤖 <b>JARVIS</b>: {last_reply.replace(chr(10), '<br>')}<br><br><span style='color:#00FFA3;font-size:13px;'>🎙️ <b>Listening for follow-up...</b></span><br>"
                    )
                    timeout_val = 8.0 if is_first_turn else FOLLOW_UP_TIMEOUT_SECONDS

                    query_text = self._stream_speech_utterance(
                        timeout=timeout_val,
                        prompt_header=prompt_header,
                    )

                    if not self.is_running:
                        break

                    if not query_text:
                        # Timeout expired without speech -> go to standby
                        if not is_first_turn:
                            formatted_prev = last_reply.replace("\n", "<br>")
                            self.update_text_signal.emit(
                                f"🤖 <b>JARVIS</b>: {formatted_prev}<br><br>"
                                "<i>Entering standby mode, sir.</i>"
                            )
                        else:
                            self.update_text_signal.emit("<i>No command detected. Returning to standby.</i>")
                        self.sleep_signal.emit()
                        in_conversation = False
                        break

                    # Check for standby/sleep keywords
                    lower_query = query_text.lower().strip()
                    if any(k in lower_query for k in [
                        "go to sleep", "goodbye", "good night", "standby", "stop listening",
                        "shut down", "that's all", "that is all", "nevermind", "never mind",
                        "thank you jarvis", "thanks jarvis"
                    ]):
                        self.update_text_signal.emit(
                            f"<b>You</b>: {query_text}<br><br>"
                            f"🤖 <b>JARVIS</b>: Standing by, sir."
                        )
                        if self.tts_engine:
                            self.tts_engine.speak("Standing by, sir.", blocking=True)
                        self.sleep_signal.emit()
                        in_conversation = False
                        break

                    # --- Phase E: Ollama Reasoning & Real-Time Speech Synthesis ---
                    result = self._generate_and_speak(query_text)

                    # 1. Handle Voice Activity Barge-In (user spoke while JARVIS was talking)
                    while isinstance(result, tuple) and result[0] == "interrupted_with_speech":
                        speech_slices = result[1]
                        if not speech_slices:
                            result = ("interrupted_wake", None)
                            break

                        query_text = self._stream_speech_utterance(
                            timeout=8.0,
                            initial_slices=speech_slices,
                            prompt_header="⚡ <b>Interrupted</b><br>",
                        )

                        print(f"[JARVIS] Heard barge-in query: '{query_text}'", flush=True)
                        if not query_text:
                            # User interrupted but no new query was spoken
                            self.update_text_signal.emit(
                                "⚡ <b>Interrupted</b><br>"
                                "<span style='color:#00FFA3;font-size:13px;'>🎙️ <b>Standing by...</b> <i>(speak your question or pause)</i></span>"
                            )
                            result = ""
                            break

                        lower_barge = query_text.lower().strip()
                        # If user commanded JARVIS to stop/be quiet
                        if any(k in lower_barge for k in ["stop", "quiet", "be quiet", "shut up", "pause", "hush", "nevermind", "never mind", "cancel"]):
                            self.update_text_signal.emit(
                                f"<b>You</b>: {query_text}<br><br>"
                                f"🤖 <b>JARVIS</b>: Standing by, sir."
                            )
                            if self.tts_engine:
                                self.tts_engine.speak("Standing by, sir.", blocking=True)
                            result = ""
                            break

                        # Answer user's new question immediately
                        result = self._generate_and_speak(query_text)

                    # 2. Handle Wake Word or Standby Barge-In
                    if isinstance(result, tuple) and result[0] in ("interrupted_wake", "interrupted_with_speech"):
                        self._flush_stream()
                        self.wake_signal.emit()
                        self.update_text_signal.emit(
                            "⚡ <b>JARVIS Activated</b><br>"
                            "<span style='color:#00FFA3;font-size:13px;'>Yes sir? Standing by for your command...</span>"
                        )
                        if self.tts_engine:
                            self.tts_engine.speak("Yes sir?", blocking=True)
                        is_first_turn = True
                        last_reply = ""
                        continue

                    elif result == "interrupted":
                        is_first_turn = True
                        last_reply = ""
                        continue

                    last_reply = result if isinstance(result, str) else ""
                    is_first_turn = False

            except Exception as e:
                self.update_text_signal.emit(f"<span style='color:#FF5555;'>Audio Loop Error: {e}</span>")
                time.sleep(1.0)

        self._cleanup()

    def _emit_partial(self, partial_text: str, prompt_header: str):
        """Emit real-time partial transcription via PyQt6 signals to dynamically update HUD word-by-word."""
        if not partial_text:
            return
        self.partial_text_signal.emit(partial_text)
        self.update_text_signal.emit(
            f"{prompt_header}"
            f"<span style='color:#FFFFFF;font-size:16px;'><b>You: </b>{partial_text} <span style='color:#00FF66;'>●</span></span>"
        )

    def _stream_speech_utterance(
        self,
        timeout: float = 8.0,
        initial_slices: Optional[list] = None,
        prompt_header: str = "🎙️ <b>Listening...</b><br>",
    ) -> str:
        """
        Real-time streaming speech capture using sherpa-onnx Zipformer transducer.
        Continuously feeds sounddevice audio slices into sherpa-onnx, emits partial
        transcription results word-by-word as you speak, and triggers instant handoff
        upon silence / endpoint detection.
        """
        self.streaming_asr.reset_stream()

        pre_roll = collections.deque(maxlen=6)
        if initial_slices:
            for s in initial_slices:
                self.streaming_asr.accept_waveform(s)
                pre_roll.append(s)

        is_speaking = bool(initial_slices)
        start_time = time.time()
        silence_start = None
        noise_floor = 120.0
        total_frames = 0
        max_frames = int(MAX_RECORDING_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

        # Show initial prompt or partial from initial chunks
        current_partial = self.streaming_asr.get_partial_text()
        if current_partial:
            self._emit_partial(current_partial, prompt_header)
        else:
            self.update_text_signal.emit(
                f"{prompt_header}<span style='color:#70FFA3;font-size:13px;'>Speak your question (pause when done)...</span>"
            )

        while self.is_running and not self.interrupted:
            # Timeout if no speech started within timeout window
            if not is_speaking and (time.time() - start_time > timeout):
                return ""

            try:
                samples, _ = self.stream.read(CHUNK_SIZE)
            except Exception:
                continue

            audio_slice = samples.flatten()
            total_frames += 1
            rms = float(np.sqrt(np.mean(audio_slice ** 2))) * 32767.0
            dynamic_threshold = max(BASE_ENERGY_THRESHOLD, noise_floor * 2.2)

            # Ingest chunk into real-time streaming Zipformer
            updated_partial = self.streaming_asr.accept_waveform(audio_slice)
            partial_text = self.streaming_asr.get_partial_text()

            # Ingest chunk into real-time streaming Zipformer ONLY after speech begins
            if not is_speaking:
                if rms > dynamic_threshold:
                    is_speaking = True
                    silence_start = None
                    self.streaming_asr.reset_stream()
                    for pr in pre_roll:
                        self.streaming_asr.accept_waveform(pr)
                    self.streaming_asr.accept_waveform(audio_slice)
                    self.update_text_signal.emit(
                        f"{prompt_header}<span style='color:#70FFA3;font-size:13px;'>🎙️ <b>Hearing you...</b></span>"
                    )
                else:
                    pre_roll.append(audio_slice)
                    noise_floor = noise_floor * 0.97 + rms * 0.03
            else:
                total_frames += 1
                updated_partial = self.streaming_asr.accept_waveform(audio_slice)
                partial_text = self.streaming_asr.get_partial_text()

                if updated_partial:
                    self._emit_partial(updated_partial, prompt_header)

                # Endpoint / pause detection
                is_endpoint = self.streaming_asr.is_endpoint()
                if rms < dynamic_threshold:
                    if silence_start is None:
                        silence_start = time.time()
                else:
                    silence_start = None

                silence_dur = (time.time() - silence_start) if silence_start else 0.0

                if (is_endpoint and bool(partial_text)) or (silence_dur >= DEFAULT_PAUSE_THRESHOLD and bool(partial_text)) or total_frames >= max_frames:
                    final_text = self.streaming_asr.get_final_text(reset=True)
                    print(f"[JARVIS Streaming ASR] Speech endpoint reached! Final query: '{final_text}'", flush=True)
                    return final_text

        return ""

    def _is_screen_query(self, text: str) -> bool:
        """Detect if the user is asking JARVIS to look at or describe their screen."""
        lower = text.lower()
        screen_triggers = [
            "screen", "look at my", "see my", "view my", "check my",
            "what am i looking at", "what's on my", "what is on my",
            "what do you see", "read my", "what's happening", "what is happening",
            "describe my", "what window", "what is displayed", "on this display",
            "look at this", "see this", "what is this"
        ]
        return any(t in lower for t in screen_triggers)

    def _capture_screen_bytes(self) -> Optional[bytes]:
        """Capture the primary monitor screenshot as compressed JPEG bytes."""
        try:
            screen = QApplication.primaryScreen()
            if not screen:
                return None
            pixmap = screen.grabWindow(0)
            if pixmap.isNull():
                return None

            # Scale to max 1280 width to optimize Vision inference latency on GPU/CPU
            if pixmap.width() > 1280:
                pixmap = pixmap.scaledToWidth(1280, Qt.TransformationMode.SmoothTransformation)

            buf = QBuffer()
            buf.open(QIODevice.OpenModeFlag.WriteOnly)
            pixmap.save(buf, "JPEG", 75)
            return bytes(buf.data())
        except Exception as e:
            print(f"[JARVIS Vision Error] Screen capture failed: {e}", flush=True)
            return None

    def _generate_and_speak(self, query_text: str):
        """Stream Ollama tokens to the HUD and synthesize audio in parallel via Kokoro TTS."""
        # Detect if user is asking about the screen
        is_screen = self._is_screen_query(query_text) and self.vision_available
        image_bytes = None

        if is_screen:
            self.update_text_signal.emit(
                f"<b>You</b>: {query_text}<br><br>"
                f"🤖 <b>JARVIS</b>: 👁️ <i>Inspecting your screen, sir...</i>"
            )
            time.sleep(0.15)
            image_bytes = self._capture_screen_bytes()
        else:
            self.update_text_signal.emit(
                f"<b>You</b>: {query_text}<br><br>"
                f"🤖 <b>JARVIS</b>: <i>Thinking...</i>"
            )

        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")

        if image_bytes:
            # Multimodal Vision reasoning with moondream
            chosen_model = self.vision_model_name
            vision_prompt = (
                "You are Jarvis, a perceptive and articulate AI assistant. Look closely at this screenshot of the user's active computer screen. "
                f"The user asked: '{query_text}'. "
                "Describe what is happening on the screen concisely in natural, spoken conversational English. "
                "Identify active applications, editor windows, terminals, websites, or errors if visible. "
                "Respond in 2 to 3 natural spoken sentences. Avoid bullet points or markdown syntax."
            )
            # Include recent chat history for context continuity during vision queries
            recent_turns = [m for m in self.chat_history[-6:] if m.get("role") in ("user", "assistant")]
            messages = recent_turns + [
                {"role": "user", "content": vision_prompt, "images": [image_bytes]}
            ]
        else:
            messages = list(self.chat_history[-DEFAULT_MAX_CHAT_HISTORY:])
            messages.append({"role": "user", "content": query_text})

        full_tokens = []
        if self.tts_engine:
            self.tts_engine.start_stream()

        self.interrupted = False
        gen_error = None

        def _gen_worker():
            nonlocal gen_error
            try:
                if image_bytes:
                    response_stream = ollama.chat(
                        model=chosen_model,
                        messages=messages,
                        stream=True,
                        options={"num_ctx": DEFAULT_NUM_CTX},
                    )

                    for chunk in response_stream:
                        if not self.is_running or self.interrupted:
                            break
                        if hasattr(chunk, "message") and hasattr(chunk.message, "content"):
                            token = chunk.message.content
                        elif isinstance(chunk, dict):
                            token = chunk.get("message", {}).get("content", "")
                        else:
                            token = getattr(chunk, "message", {}).get("content", "")

                        if token:
                            full_tokens.append(token)
                            if self.tts_engine and not self.interrupted:
                                self.tts_engine.feed_stream(token)

                            partial_text = "".join(full_tokens)
                            html_formatted = partial_text.replace("\n", "<br>")
                            self.update_text_signal.emit(
                                f"<b>You</b>: {query_text}<br><br>"
                                f"🤖 <b>JARVIS</b>: {html_formatted}"
                            )
                else:
                    def _on_chunk(token: str):
                        if not self.is_running or self.interrupted:
                            return
                        full_tokens.append(token)
                        if self.tts_engine and not self.interrupted:
                            self.tts_engine.feed_stream(token)
                        partial_text = "".join(full_tokens)
                        html_formatted = partial_text.replace("\n", "<br>")
                        self.update_text_signal.emit(
                            f"<b>You</b>: {query_text}<br><br>"
                            f"🤖 <b>JARVIS</b>: {html_formatted}"
                        )

                    for _ in self.ollama_engine.stream_chat(
                        messages=messages,
                        on_chunk=_on_chunk,
                    ):
                        if not self.is_running or self.interrupted:
                            break

                if self.tts_engine and not self.interrupted:
                    self.tts_engine.end_stream(wait=False)
            except Exception as e:
                gen_error = e

        gen_thread = threading.Thread(target=_gen_worker, daemon=True)
        gen_thread.start()

        # Flush any stale audio buffered during transcription/screen capture so barge-in has 0ms latency
        self._flush_stream()

        # --- Active Interruption / Voice Activity Barge-In Audio Loop ---
        # While the LLM is generating or TTS is speaking audio, actively monitor mic via sounddevice!
        pre_roll_interruption = collections.deque(maxlen=6)
        consecutive_speech_chunks = 0
        interrupted_speech_slices = None
        noise_floor = 60.0

        while self.is_running and (gen_thread.is_alive() or (self.tts_engine and self.tts_engine.is_speaking())):
            if self.interrupted:
                break

            try:
                samples, _ = self.stream.read(CHUNK_SIZE)
            except Exception:
                time.sleep(0.02)
                continue

            audio_slice = samples.flatten()
            rms = float(np.sqrt(np.mean(audio_slice ** 2))) * 32767.0
            pre_roll_interruption.append(audio_slice)

            # 1. Wake word barge-in check
            wake_interrupted = False
            pcm_bytes = (audio_slice * 32767.0).astype(np.int16).tobytes()
            if self.wakeword_engine and self.wakeword_engine.process_pcm_chunk(pcm_bytes):
                last_score = self.wakeword_engine.get_last_score()
                print(f"[JARVIS] Interruption detected via wake word! (Score: {last_score:.3f}) Stopping speech...", flush=True)
                wake_interrupted = True

            # 2. Voice Activity Barge-In: user simply begins speaking while JARVIS is talking!
            # Only trigger voice barge-in when TTS is actively speaking audio out loud!
            is_tts_playing = bool(self.tts_engine and self.tts_engine.is_speaking())
            if is_tts_playing:
                barge_in_threshold = max(380.0, noise_floor * 2.4)
                if rms > barge_in_threshold:
                    consecutive_speech_chunks += 1
                    # 2 consecutive chunks (~160ms) of sustained human vocalization
                    if consecutive_speech_chunks >= 2:
                        print(f"[JARVIS] User speech barge-in detected! (RMS: {rms:.1f}) Stopping speech immediately...", flush=True)
                        self.interrupted = True
                        interrupted_speech_slices = list(pre_roll_interruption)
                        break
                else:
                    consecutive_speech_chunks = 0
                    noise_floor = noise_floor * 0.95 + min(rms, 250.0) * 0.05
            else:
                consecutive_speech_chunks = 0
                noise_floor = noise_floor * 0.95 + min(rms, 250.0) * 0.05

            if wake_interrupted:
                self.interrupted = True
                break

        if self.interrupted:
            # STOP TTS AND OLLAMA IMMEDIATELY
            if self.tts_engine:
                self.tts_engine.stop()
            self.ollama_engine.stop_generation()

            if interrupted_speech_slices:
                # The user interrupted by speaking! Pass slices to caller for real-time streaming
                self.update_text_signal.emit(
                    "⚡ <b>Interrupted</b><br>"
                    "<span style='color:#00FFA3;'>🎙️ <b>Hearing you...</b></span>"
                )
                return ("interrupted_with_speech", interrupted_speech_slices)
            else:
                self.update_text_signal.emit(
                    f"<b>You</b>: {query_text}<br><br>"
                    f"🤖 <b>JARVIS</b>: <span style='color:#FFB800;'><i>[Interrupted]</i></span>"
                )
                return ("interrupted_wake", None)

        if gen_error:
            if self.tts_engine:
                self.tts_engine.stop()
            self.update_text_signal.emit(
                f"<b>You</b>: {query_text}<br><br>"
                f"<span style='color:#FF5555;'>[Ollama Error: {gen_error}]</span>"
            )
            return ""

        full_response = "".join(full_tokens)
        self.chat_history.append({"role": "user", "content": query_text})
        self.chat_history.append({"role": "assistant", "content": full_response})
        return full_response

    def _compute_rms_raw(self, audio_data) -> float:
        """Compute RMS amplitude directly from float32 array or 16-bit PCM buffer."""
        if isinstance(audio_data, np.ndarray):
            return float(np.sqrt(np.mean(audio_data ** 2))) * 32767.0
        count = len(audio_data) // 2
        if count == 0:
            return 0.0
        try:
            shorts = struct.unpack(f"{count}h", audio_data)
            sum_sq = sum(s * s for s in shorts)
            return math.sqrt(sum_sq / count)
        except Exception:
            return 0.0

    def _flush_stream(self):
        """Purge any audio accumulated in sounddevice buffer while assistant was speaking."""
        if self.stream:
            try:
                avail = self.stream.read_available
                if avail > 0:
                    self.stream.read(avail)
            except Exception:
                pass

    def stop(self):
        """Signal thread to stop and terminate all playback."""
        self.is_running = False
        if self.tts_engine:
            self.tts_engine.stop()

    def _cleanup(self):
        """Release audio devices safely."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None


# =============================================================================
# The HUD (QMainWindow)
# =============================================================================
class JarvisHUD(QMainWindow):
    """
    Transparent, frameless, always-on-top HUD window for JARVIS.
    Features:
    - Completely hidden on startup.
    - Displays at the bottom-center of the user's primary monitor.
    - Single styled QLabel (neon green, dark translucent background, rounded corners).
    - Smooth opacity fade out after 4-second readability pause.
    """
    interrupt_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)

        # 1. Window Flags: Frameless, Always On Top, Tool (hidden from taskbar)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.Tool
        )

        # 2. Transparent Window Background
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setStyleSheet("QMainWindow { background: transparent; }")

        # 3. Single styled QLabel displaying transcription & AI responses
        self.label = QLabel(self)
        self.label.setWordWrap(True)
        self.label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.label.setTextFormat(Qt.TextFormat.RichText)

        # Neon Green & Dark Translucent Styling with Rounded Corners
        self.label.setStyleSheet("""
            QLabel {
                color: #00FF66;
                background-color: rgba(10, 16, 26, 0.90);
                border: 2px solid rgba(0, 255, 102, 0.65);
                border-radius: 18px;
                padding: 18px 28px;
                font-family: 'Segoe UI', 'Consolas', 'Lucida Console', sans-serif;
                font-size: 15px;
                font-weight: 500;
                line-height: 1.45;
            }
        """)

        # Futuristic glowing drop shadow effect
        glow = QGraphicsDropShadowEffect(self.label)
        glow.setBlurRadius(28)
        glow.setColor(QColor(0, 255, 102, 140))
        glow.setOffset(0, 0)
        self.label.setGraphicsEffect(glow)

        self.setCentralWidget(self.label)

        # 4. Smooth Fade-Out Animation
        self.fade_anim = QPropertyAnimation(self, b"windowOpacity")
        self.fade_anim.setDuration(350)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.fade_anim.finished.connect(self._on_fade_finished)

        # 5. Non-blocking QTimer for 4-second readability pause before hiding
        self.sleep_timer = QTimer(self)
        self.sleep_timer.setSingleShot(True)
        self.sleep_timer.timeout.connect(self._start_fade_out)

        # 6. Start completely hidden (invisible on startup)
        self.hide()

    def reposition_to_bottom_center(self):
        """Calculate primary screen geometry and center the HUD at the bottom."""
        screen = QApplication.primaryScreen()
        if not screen:
            return
        geo = screen.availableGeometry()

        self.label.adjustSize()
        hint = self.label.sizeHint()

        max_w = min(1100, int(geo.width() * 0.90))
        min_w = 480
        target_w = max(min_w, min(hint.width() + 48, max_w))
        target_h = min(int(geo.height() * 0.70), max(72, hint.height() + 24))

        self.resize(target_w, target_h)

        # Bottom-center positioning (45px above primary monitor taskbar)
        x = geo.x() + (geo.width() - target_w) // 2
        y = geo.y() + geo.height() - target_h - 45
        self.move(x, y)

    def on_wake(self):
        """Slot for wake_signal -> Instantly reveal HUD window."""
        self.sleep_timer.stop()
        self.fade_anim.stop()
        self.setWindowOpacity(1.0)
        self.reposition_to_bottom_center()
        self.show()
        self.raise_()
        self.activateWindow()

    def on_update_text(self, text: str):
        """Slot for update_text_signal -> Update QLabel and readjust geometry."""
        self.sleep_timer.stop()
        self.fade_anim.stop()
        self.setWindowOpacity(1.0)
        self.label.setText(text)
        self.reposition_to_bottom_center()
        if not self.isVisible():
            self.show()
            self.raise_()

    def on_partial_text(self, partial_text: str):
        """Slot for partial_text_signal -> Direct hook for real-time word-by-word streaming ASR."""
        # Active HUD updates are also dispatched via update_text_signal with rich HTML formatting.
        pass

    def on_sleep(self):
        """Slot for sleep_signal -> Pause for 4 seconds so user can read, then hide."""
        self.sleep_timer.stop()
        self.sleep_timer.start(4000)

    def _start_fade_out(self):
        """Initiate smooth opacity fade-out."""
        self.fade_anim.stop()
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()

    def _on_fade_finished(self):
        """Hide window once opacity reaches zero, then reset opacity."""
        if self.windowOpacity() <= 0.05:
            self.hide()
            self.setWindowOpacity(1.0)

    def keyPressEvent(self, event):
        """Allow user to manually dismiss HUD and interrupt speech with Escape key."""
        if event.key() == Qt.Key.Key_Escape:
            self.interrupt_requested.emit()
            self.hide()
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        """Clicking anywhere on HUD halts speech output and dismisses."""
        self.interrupt_requested.emit()
        self.hide()
        super().mousePressEvent(event)


# =============================================================================
# Helper: Dynamic System Tray Icon Generation
# =============================================================================
def create_tray_icon() -> QIcon:
    """Generate a crisp glowing JARVIS arc-reactor icon for Windows system tray."""
    pixmap = QPixmap(32, 32)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    # Outer dark circle
    painter.setBrush(QBrush(QColor(10, 20, 28)))
    painter.setPen(QPen(QColor(0, 255, 102), 2))
    painter.drawEllipse(2, 2, 28, 28)

    # Inner glowing core
    painter.setBrush(QBrush(QColor(0, 255, 102)))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(12, 12, 8, 8)

    # Accent ticks
    painter.setPen(QPen(QColor(0, 255, 102, 180), 1))
    painter.drawLine(16, 4, 16, 8)
    painter.drawLine(16, 24, 16, 28)
    painter.drawLine(4, 16, 8, 16)
    painter.drawLine(24, 16, 28, 16)

    painter.end()
    return QIcon(pixmap)


# =============================================================================
# Application Entry Point
# =============================================================================
def main():
    # Enforce single running instance to avoid microphone device conflicts
    import socket
    global _lock_socket
    _lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        _lock_socket.bind(("127.0.0.1", 47823))
        _lock_socket.listen(2)
    except OSError:
        # An instance is already running! Connect and tell it to show the HUD window
        try:
            client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client.connect(("127.0.0.1", 47823))
            client.sendall(b"show")
            client.close()
        except Exception:
            pass
        return

    app = QApplication(sys.argv)
    app.setApplicationName("JARVIS Voice Assistant")
    # Critical: Do not quit application when HUD window hides
    app.setQuitOnLastWindowClosed(False)

    hud = JarvisHUD()

    class _InstanceMessenger(QObject):
        show_signal = pyqtSignal()

    messenger = _InstanceMessenger()
    def _on_secondary_show():
        hud.on_wake()
        hud.on_update_text(
            "⚡ <b>JARVIS is active and listening</b><br>"
            "<span style='color:#00FFA3;font-size:13px;'>Standing by for your command. Say <b>\"Hey Jarvis\"</b></span>"
        )
        hud.on_sleep()

    messenger.show_signal.connect(_on_secondary_show)

    def _socket_ipc_worker():
        while True:
            try:
                conn, _ = _lock_socket.accept()
                data = conn.recv(64)
                conn.close()
                if b"show" in data:
                    messenger.show_signal.emit()
            except Exception:
                break

    threading.Thread(target=_socket_ipc_worker, daemon=True).start()

    # Determine command-line options
    wake_required = True
    args = [a.lower() for a in sys.argv[1:]]
    if "--live" in args or "-l" in args or "--no-wake" in args:
        wake_required = False

    # Initialize and connect background audio thread
    listener_thread = AudioListenerThread(wake_word_required=wake_required)
    listener_thread.wake_signal.connect(hud.on_wake)
    listener_thread.update_text_signal.connect(hud.on_update_text)
    listener_thread.partial_text_signal.connect(hud.on_partial_text)
    listener_thread.sleep_signal.connect(hud.on_sleep)
    hud.interrupt_requested.connect(listener_thread.trigger_interrupt)

    # Windows System Tray Integration
    tray_icon = QSystemTrayIcon(create_tray_icon(), app)
    tray_menu = QMenu()

    status_header = QAction("⚡ JARVIS Background Service", tray_menu)
    status_header.setEnabled(False)
    tray_menu.addAction(status_header)
    tray_menu.addSeparator()

    def test_wake_action():
        """Simulate wake word event from tray icon for immediate verification."""
        hud.on_wake()
        hud.on_update_text(
            "⚡ <b>JARVIS HUD Test</b><br>"
            "<span style='color:#00FFA3;'>HUD triggered successfully via System Tray!</span>"
        )
        hud.on_sleep()

    test_action = QAction("Trigger Wake Word (Test HUD)", tray_menu)
    test_action.triggered.connect(test_wake_action)
    tray_menu.addAction(test_action)

    def screen_vision_action():
        """Trigger instant screen analysis from the system tray."""
        hud.on_wake()
        hud.on_update_text("⚡ <b>JARVIS Vision</b><br>👁️ <i>Inspecting your screen...</i>")
        def _vision_worker():
            listener_thread._generate_and_speak("What is happening on my screen right now?")
            hud.on_sleep()
        threading.Thread(target=_vision_worker, daemon=True).start()

    vision_action = QAction("👁️ Look at Screen & Describe", tray_menu)
    vision_action.triggered.connect(screen_vision_action)
    tray_menu.addAction(vision_action)

    def toggle_hud_action():
        if hud.isVisible():
            hud.hide()
        else:
            hud.on_wake()
            hud.on_update_text("⚡ <b>JARVIS HUD</b><br>Standing by for commands...")

    toggle_action = QAction("Toggle HUD Visibility", tray_menu)
    toggle_action.triggered.connect(toggle_hud_action)
    tray_menu.addAction(toggle_action)

    stop_speech_action = QAction("🛑 Stop Speaking (Interrupt)", tray_menu)
    stop_speech_action.triggered.connect(listener_thread.trigger_interrupt)
    tray_menu.addAction(stop_speech_action)

    tray_menu.addSeparator()
    quit_action = QAction("Exit JARVIS", tray_menu)
    quit_action.triggered.connect(app.quit)
    tray_menu.addAction(quit_action)

    tray_icon.setContextMenu(tray_menu)
    tray_icon.setToolTip("JARVIS Voice Assistant (Listening in background)")
    tray_icon.show()

    # Clean shutdown on application quit
    def on_exit():
        listener_thread.stop()
        listener_thread.wait(2000)

    app.aboutToQuit.connect(on_exit)

    # Visual startup confirmation: flash HUD briefly so user immediately sees JARVIS is active
    hud.on_wake()
    hud.on_update_text(
        "⚡ <b>JARVIS Online</b><br>"
        "<span style='color:#00FFA3;font-size:13px;'>Background service active. Say \"Hey Jarvis\" to interact.</span>"
    )
    hud.on_sleep()

    try:
        tray_icon.showMessage(
            "JARVIS Online",
            "Background service active. Say 'Hey Jarvis' or click icon to interact.",
            QSystemTrayIcon.MessageIcon.Information,
            3500
        )
    except Exception:
        pass

    # Launch background listening thread
    listener_thread.start()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
