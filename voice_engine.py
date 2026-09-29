"""
Voice Engine for JARVIS Voice Assistant.
========================================
Architecture:
- Real-time continuous streaming microphone capture via sounddevice (sd.InputStream)
- Low-latency ONNX streaming Zipformer transducer via sherpa-onnx (no batch transcribers)
- openWakeWord local "Hey Jarvis" detection
- Real-time word-by-word token emissions and zero-latency endpoint handoff
"""

import collections
import threading
import time
from typing import Callable, List, Optional, Tuple
import numpy as np
import sounddevice as sd

from config import (
    BASE_ENERGY_THRESHOLD,
    CHANNELS,
    CHUNK_SIZE,
    DEFAULT_PAUSE_THRESHOLD,
    DEFAULT_STREAMING_MODEL,
    MAX_RECORDING_SECONDS,
    MIN_SPEECH_DURATION,
    SAMPLE_RATE,
    STREAMING_ENDPOINT_RULE1_SILENCE,
    STREAMING_ENDPOINT_RULE2_SILENCE,
    STREAMING_ENDPOINT_RULE3_MAX_LEN,
    STREAMING_NUM_THREADS,
)
from streaming_asr import StreamingASREngine
from wakeword_engine import WakeWordEngine


class VoiceEngine:
    """
    Continuous streaming voice engine for JARVIS.
    Captures audio slices directly through sounddevice and decodes them in real time
    with sherpa-onnx.
    """

    def __init__(
        self,
        streaming_model_name: str = DEFAULT_STREAMING_MODEL,
        **kwargs,
    ):
        self.use_streaming = True
        self.wakeword_engine = WakeWordEngine()

        # Real-time Streaming ASR Engine (Zipformer transducer via sherpa-onnx)
        self.streaming_model_name = streaming_model_name
        self.streaming_engine = StreamingASREngine(
            model_name=streaming_model_name,
            num_threads=STREAMING_NUM_THREADS,
            sample_rate=SAMPLE_RATE,
            rule1_silence=STREAMING_ENDPOINT_RULE1_SILENCE,
            rule2_silence=STREAMING_ENDPOINT_RULE2_SILENCE,
            rule3_max_utterance=STREAMING_ENDPOINT_RULE3_MAX_LEN,
        )

        # Continuous Listener controls
        self.live_running = False
        self.live_paused = False
        self._live_thread: Optional[threading.Thread] = None
        self._live_stop_event = threading.Event()
        self.pause_threshold = DEFAULT_PAUSE_THRESHOLD
        self.current_device_index: Optional[int] = None
        self.waiting_for_wakeword = False
        self.barge_in_enabled = False
        self.consecutive_barge_chunks = 0

    def set_barge_in(self, enabled: bool):
        """Enable or disable voice energy barge-in interruption."""
        self.barge_in_enabled = enabled
        self.consecutive_barge_chunks = 0

    def get_input_devices(self) -> List[Tuple[Optional[int], str]]:
        """Return list of available audio input devices (index, name) via sounddevice."""
        devices = []
        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev.get("max_input_channels", 0) > 0:
                    name = dev.get("name", f"Device {i}").strip()
                    devices.append((i, name))
        except Exception:
            devices = [(None, "Default System Microphone")]
        return devices if devices else [(None, "Default System Microphone")]

    def load_streaming_asr(self):
        """Initialize the real-time streaming Zipformer ASR model."""
        self.streaming_engine.load()

    def load_wakeword_model(self):
        """Initialize openWakeWord model."""
        self.wakeword_engine.load()

    # =========================================================================
    # Continuous Audio Loop with sounddevice & Streaming ASR
    # =========================================================================
    def start_live_mode(
        self,
        device_index: Optional[int] = None,
        use_wakeword: bool = False,
        pause_threshold: float = DEFAULT_PAUSE_THRESHOLD,
        on_wakeword: Optional[Callable[[], None]] = None,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_audio_level: Optional[Callable[[float], None]] = None,
        on_partial_transcription: Optional[Callable[[str], None]] = None,
        on_endpoint_detected: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ):
        """
        Start continuous listener stream via sounddevice.
        - Keeps a continuous sounddevice microphone stream open.
        - Passes tiny audio slices (around 50ms - 100ms) directly into sherpa-onnx.
        - Emits partial transcription results word-by-word with virtually zero latency.
        - Fires on_endpoint_detected immediately upon silence detection.
        """
        if self.live_running:
            return

        self.streaming_engine.load()

        self.live_running = True
        self.live_paused = False
        self.waiting_for_wakeword = use_wakeword
        self.pause_threshold = pause_threshold
        self.current_device_index = device_index
        self._live_stop_event.clear()

        def _live_worker():
            stream = None
            try:
                # Open continuous sounddevice InputStream
                stream = sd.InputStream(
                    samplerate=SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="float32",
                    blocksize=CHUNK_SIZE,
                    device=self.current_device_index,
                )
                stream.start()

                pre_roll_chunks = collections.deque(maxlen=6)
                is_speaking = False
                silence_start_time = None
                noise_floor = 150.0
                total_frames = 0
                max_frames = int(MAX_RECORDING_SECONDS * SAMPLE_RATE / CHUNK_SIZE)

                while not self._live_stop_event.is_set():
                    if self.live_paused:
                        # Flush any queued audio while paused
                        try:
                            avail = stream.read_available
                            if avail > 0:
                                stream.read(avail)
                        except Exception:
                            pass
                        time.sleep(0.04)
                        if on_audio_level:
                            on_audio_level(0.0)
                        continue

                    try:
                        samples, overflowed = stream.read(CHUNK_SIZE)
                    except Exception:
                        continue

                    # samples is (CHUNK_SIZE, 1) float32 array
                    audio_slice = samples.flatten()

                    # Compute RMS amplitude
                    rms = float(np.sqrt(np.mean(audio_slice ** 2))) * 32767.0
                    normalized_lvl = min(1.0, max(0.0, (rms - 80.0) / 3200.0))

                    if on_audio_level:
                        on_audio_level(normalized_lvl)

                    # 1. WAKE WORD DETECTION & BARGE-IN PHASE
                    if self.waiting_for_wakeword:
                        pre_roll_chunks.append(audio_slice)
                        pcm_bytes = (audio_slice * 32767.0).astype(np.int16).tobytes()

                        # 2. REAL-TIME STREAMING ASR PHASE (sherpa-onnx)
                    dynamic_threshold = max(BASE_ENERGY_THRESHOLD, noise_floor * 2.2)

                    if not is_speaking:
                        # DO NOT feed silent/fan noise into the neural transducer!
                        if rms > dynamic_threshold:
                            is_speaking = True
                            silence_start_time = None
                            total_frames = 0
                            # Reset the stream cleanly right before speaking starts
                            self.streaming_engine.reset_stream()
                            for pr in pre_roll_chunks:
                                self.streaming_engine.accept_waveform(pr)
                            self.streaming_engine.accept_waveform(audio_slice)
                            if on_speech_start:
                                on_speech_start()
                        else:
                            pre_roll_chunks.append(audio_slice)
                            noise_floor = noise_floor * 0.97 + rms * 0.03
                    else:
                        total_frames += 1
                        # Actively feed speech frames and emit live tokens
                        updated_partial = self.streaming_engine.accept_waveform(audio_slice)
                        current_partial = self.streaming_engine.get_partial_text()

                        if updated_partial and on_partial_transcription:
                            on_partial_transcription(updated_partial)

                        # Endpoint / pause detection
                        is_endpoint_hit = self.streaming_engine.is_endpoint()
                        if rms < dynamic_threshold:
                            if silence_start_time is None:
                                silence_start_time = time.time()
                        else:
                            silence_start_time = None

                        silence_dur = (time.time() - silence_start_time) if silence_start_time else 0.0

                        # Handoff condition: endpoint reached or silence threshold met
                        if (is_endpoint_hit and bool(current_partial)) or (silence_dur >= self.pause_threshold and bool(current_partial)) or total_frames >= max_frames:
                            final_query = self.streaming_engine.get_final_text(reset=True)
                            is_speaking = False
                            silence_start_time = None
                            self.live_paused = True

                            if on_endpoint_detected and final_query:
                                on_endpoint_detected(final_query)

                            pre_roll_chunks.clear()
                            continue

                    # Emit partial transcription word-by-word dynamically
                    if updated_partial and on_partial_transcription:
                        on_partial_transcription(updated_partial)

                    # 3. ENDPOINT DETECTION (The Handoff)
                    if is_speaking:
                        is_endpoint_hit = self.streaming_engine.is_endpoint()

                        if rms < dynamic_threshold:
                            if silence_start_time is None:
                                silence_start_time = time.time()
                        else:
                            silence_start_time = None

                        silence_dur = (time.time() - silence_start_time) if silence_start_time else 0.0

                        # Instant handoff condition
                        if (is_endpoint_hit and bool(current_partial)) or (silence_dur >= self.pause_threshold and bool(current_partial)) or total_frames >= max_frames:
                            final_query = self.streaming_engine.get_final_text(reset=True)
                            is_speaking = False
                            silence_start_time = None
                            self.live_paused = True

                            if on_endpoint_detected and final_query:
                                on_endpoint_detected(final_query)

                            pre_roll_chunks.clear()
                            continue

            except Exception as e:
                if on_error and not self._live_stop_event.is_set():
                    on_error(f"Audio stream error: {e}")
            finally:
                if stream:
                    try:
                        stream.stop()
                        stream.close()
                    except Exception:
                        pass
                self.live_running = False

        self._live_thread = threading.Thread(target=_live_worker, daemon=True)
        self._live_thread.start()

    def pause_live_mode(self):
        self.live_paused = True

    def resume_live_mode(self, wait_for_wakeword: bool = False):
        self.waiting_for_wakeword = wait_for_wakeword
        self.wakeword_engine.reset()
        self.streaming_engine.reset_stream()
        self.live_paused = False

    def stop_live_mode(self):
        self._live_stop_event.set()
        self.live_running = False
        self.live_paused = False

    def cleanup(self):
        self.stop_live_mode()
