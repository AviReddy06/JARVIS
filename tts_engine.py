"""
Kokoro Real-Time Micro-Streaming TTS Engine for JARVIS Voice Assistant.
Features:
- Instant first-phrase synthesis (triggers on first 3-4 words or first comma)
- Decoupled two-stage pipeline: Synthesis thread + Playback thread in parallel
- Zero gap between spoken sentences
- Instant interruption and playback cancellation
"""

import os
import queue
import re
import threading
import time
from typing import Optional
import kokoro_onnx
import sounddevice as sd

from config import DEFAULT_VOICE, DEFAULT_VOICE_SPEED


class TTSEngine:
    def __init__(
        self,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
        speed: float = DEFAULT_VOICE_SPEED,
    ):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_dir, "models", "kokoro-v1.0.onnx")
        self.voices_path = voices_path or os.path.join(base_dir, "models", "voices-v1.0.bin")
        self.voice = voice
        self.speed = speed

        self.kokoro: Optional[kokoro_onnx.Kokoro] = None
        self._stop_event = threading.Event()

        # Two-stage pipeline queues
        self._text_queue = queue.Queue()    # Receives text chunks to synthesize
        self._audio_queue = queue.Queue()   # Receives (samples, sample_rate) to play

        self._text_buffer = ""
        self._is_first_chunk = True
        self._is_speaking = False
        self._stream_active = False
        self._is_synthesizing = False
        self._is_playing = False

        self._workers_running = False
        self._synth_thread: Optional[threading.Thread] = None
        self._play_thread: Optional[threading.Thread] = None

    def load(self):
        """Initialize Kokoro model and launch two-stage pipeline threads."""
        if self.kokoro is not None:
            return

        if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
            models_dir = os.path.dirname(self.model_path)
            os.makedirs(models_dir, exist_ok=True)
            from urllib.request import urlretrieve
            if not os.path.exists(self.model_path):
                print(f"[JARVIS TTS] Downloading Kokoro model weights...", flush=True)
                urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx", self.model_path)
            if not os.path.exists(self.voices_path):
                print(f"[JARVIS TTS] Downloading Kokoro voices...", flush=True)
                urlretrieve("https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin", self.voices_path)

        if not os.path.exists(self.model_path) or not os.path.exists(self.voices_path):
            raise FileNotFoundError(
                f"Kokoro model files not found at {self.model_path} or {self.voices_path}"
            )

        self.kokoro = kokoro_onnx.Kokoro(self.model_path, self.voices_path)
        # Warm up engine
        try:
            self.kokoro.create("Jarvis", voice=self.voice, speed=self.speed)
        except Exception:
            pass

        self._start_pipeline()

    def _start_pipeline(self):
        """Start parallel synthesis and playback workers."""
        if self._workers_running:
            return
        self._workers_running = True

        self._synth_thread = threading.Thread(target=self._synth_worker, daemon=True)
        self._synth_thread.start()

        self._play_thread = threading.Thread(target=self._play_worker, daemon=True)
        self._play_thread.start()

    def _synth_worker(self):
        """Stage 1: Synthesizes text segments to audio in background."""
        while self._workers_running:
            try:
                item = self._text_queue.get(timeout=0.08)
                if item is None:
                    # Sentinel
                    self._audio_queue.put(None)
                    self._text_queue.task_done()
                    continue

                if self._stop_event.is_set():
                    self._text_queue.task_done()
                    continue

                clean = self._clean_for_speech(item)
                if clean and self.kokoro is not None:
                    try:
                        self._is_synthesizing = True
                        samples, sample_rate = self.kokoro.create(
                            clean,
                            voice=self.voice,
                            speed=self.speed
                        )
                        if not self._stop_event.is_set():
                            self._audio_queue.put((samples, sample_rate))
                    except Exception as e:
                        print(f"\n[TTS Synth Error] {e}")
                    finally:
                        self._is_synthesizing = False

                self._text_queue.task_done()

            except queue.Empty:
                continue

    def _play_worker(self):
        """Stage 2: Plays synthesized audio out loud in real time."""
        while self._workers_running:
            try:
                item = self._audio_queue.get(timeout=0.08)
                if item is None:
                    self._audio_queue.task_done()
                    self._is_speaking = False
                    continue

                if self._stop_event.is_set():
                    self._audio_queue.task_done()
                    continue

                self._is_speaking = True
                self._is_playing = True
                samples, sample_rate = item

                try:
                    sd.play(samples, samplerate=sample_rate)
                    sd.wait()
                except Exception as e:
                    print(f"\n[TTS Play Error] {e}")
                finally:
                    self._is_playing = False
                    self._audio_queue.task_done()

            except queue.Empty:
                if self._text_queue.empty() and self._audio_queue.empty():
                    self._is_speaking = False
                continue

    # =========================================================================
    # Micro-Streaming Feed
    # =========================================================================
    def start_stream(self):
        """Reset and prepare for incoming streaming tokens."""
        self.stop()
        self._stop_event.clear()
        self._text_buffer = ""
        self._is_first_chunk = True
        self._stream_active = True

    def feed_stream(self, token: str):
        """
        Feed each token directly from Ollama.
        Extracts micro-chunks so speech starts on the very first 3-4 words!
        """
        if self._stop_event.is_set():
            return

        self._text_buffer += token

        # Check if we have a speakable phrase
        phrase, remaining = self._extract_phrase(self._text_buffer, is_first=self._is_first_chunk)
        if phrase:
            self._is_first_chunk = False
            self._text_buffer = remaining
            self._text_queue.put(phrase)

    def end_stream(self, wait: bool = True):
        """Flush any remaining text in buffer and optionally wait for speech playback."""
        self._stream_active = False
        if self._text_buffer.strip() and not self._stop_event.is_set():
            self._text_queue.put(self._text_buffer.strip())
            self._text_buffer = ""

        if wait:
            self._text_queue.join()
            self._audio_queue.join()
            self._is_speaking = False

    def is_speaking(self) -> bool:
        return (
            self._stream_active
            or self._is_synthesizing
            or self._is_playing
            or self._is_speaking
            or not self._text_queue.empty()
            or not self._audio_queue.empty()
            or bool(self._text_buffer.strip())
        )

    def stop(self):
        """Immediately abort speech and purge all synthesis and playback queues."""
        self._stop_event.set()
        self._stream_active = False
        try:
            sd.stop()
        except Exception:
            pass

        # Purge text queue
        while not self._text_queue.empty():
            try:
                self._text_queue.get_nowait()
                self._text_queue.task_done()
            except Exception:
                break

        # Purge audio queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
            except Exception:
                break

        self._text_buffer = ""
        self._is_synthesizing = False
        self._is_playing = False
        self._is_speaking = False

    def speak(self, text: str, voice: Optional[str] = None, speed: Optional[float] = None, blocking: bool = True):
        """Direct one-shot speak (used for greetings and quick replies)."""
        if not text or not text.strip():
            return

        self.load()
        self.stop()
        self._stop_event.clear()

        clean_text = self._clean_for_speech(text)
        if not clean_text:
            return

        selected_voice = voice or self.voice
        selected_speed = speed or self.speed

        def _worker():
            self._is_speaking = True
            self._is_synthesizing = True
            try:
                samples, sample_rate = self.kokoro.create(
                    clean_text,
                    voice=selected_voice,
                    speed=selected_speed
                )
                self._is_synthesizing = False
                if not self._stop_event.is_set():
                    self._is_playing = True
                    sd.play(samples, samplerate=sample_rate)
                    sd.wait()
            except Exception as e:
                print(f"\n[TTS Error] {e}")
            finally:
                self._is_synthesizing = False
                self._is_playing = False
                self._is_speaking = False

        if blocking:
            _worker()
        else:
            t = threading.Thread(target=_worker, daemon=True)
            t.start()

    def _extract_phrase(self, buffer: str, is_first: bool):
        """
        Fast First Trigger:
        On the very first chunk, emit audio on the first comma OR as soon as 3 words arrive.
        On later chunks, wait for complete sentences or clauses for natural prosody.
        """
        stripped = buffer.strip()
        words = stripped.split()

        # --- 1. INSTANT FIRST PHRASE SYNTHESIS ---
        if is_first:
            # Trigger immediately if the LLM emits a greeting break (e.g., "Certainly,", "Sure,", "Hello,")
            first_punct = re.search(r"([,;:\n]+(\s+|$))", buffer)
            if first_punct and len(words) >= 1:
                idx = first_punct.end()
                phrase = buffer[:idx].strip()
                remaining = buffer[idx:]
                if phrase:
                    return phrase, remaining

            # Or trigger immediately once 3 words arrive
            if len(words) >= 3:
                # Find the boundary of the 3rd word
                match = re.search(r"^(\s*\S+\s+\S+\s+\S+)(\s+|$)", buffer)
                if match:
                    idx = match.end()
                    phrase = buffer[:idx].strip()
                    remaining = buffer[idx:]
                    return phrase, remaining

        # --- 2. SUBSEQUENT CHUNKS: NATURAL SENTENCE & CLAUSE BOUNDARIES ---
        sentence_match = re.search(r"([.!?]+(\s+|$))|(\n+)", buffer)
        if sentence_match:
            idx = sentence_match.end()
            phrase = buffer[:idx].strip()
            remaining = buffer[idx:]
            # Avoid splitting on common abbreviations
            if not re.search(r"\b(mr|mrs|ms|dr|prof|sr|jr|vs|approx|e\.g|i\.e)\.\s*$", phrase, re.IGNORECASE):
                if phrase:
                    return phrase, remaining

        # Secondary clause break for long sentences (5+ words ending with comma/semicolon)
        if len(words) >= 5 and any(c in buffer for c in [",", ";", ":", "—"]):
            clause_match = re.search(r"([,:;—](\s+|$))", buffer)
            if clause_match:
                idx = clause_match.end()
                phrase = buffer[:idx].strip()
                remaining = buffer[idx:]
                if phrase:
                    return phrase, remaining

        return None, buffer

    def _clean_for_speech(self, text: str) -> str:
        """Strip markdown syntax and format speech tokens for natural pronunciation."""
        t = re.sub(r"```[\s\S]*?```", " ", text)
        t = re.sub(r"`([^`]+)`", r"\1", t)
        t = re.sub(r"[*_~#]", "", t)
        t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)
        # Expand common acronyms for natural vocalization
        t = re.sub(r"\bAI\b", "A.I.", t)
        t = re.sub(r"\be\.g\.", "for example", t)
        t = re.sub(r"\bi\.e\.", "that is", t)
        t = re.sub(r"\bvs\.\b", "versus", t)
        # Remove emojis and strange symbols that cause robotic pauses
        t = re.sub(r"[^\w\s.,!?:;'\"]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t
