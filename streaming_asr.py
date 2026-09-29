"""
Streaming ASR Engine for JARVIS Voice Assistant.
================================================
Powered by sherpa-onnx using an ONNX streaming Zipformer transducer model.
Features:
- Real-time frame-by-frame streaming ASR (50ms - 100ms frames)
- Emits partial transcription results word-by-word with sub-millisecond CPU inference
- Real-time endpointing (silence / speech pause detection) for zero-latency LLM handoff
- Natural capitalization and grammar normalization
"""

import os
import re
import sys
from typing import Optional, Tuple
import numpy as np

try:
    import sherpa_onnx
    HAS_SHERPA_ONNX = True
except ImportError:
    HAS_SHERPA_ONNX = False

DEFAULT_MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-2023-06-26"
FALLBACK_MODEL_NAME = "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17"


def normalize_asr_text(text: str) -> str:
    """
    Format raw ASR tokens into clean, naturally capitalized spoken English.
    Handles all-caps text, sentence boundaries, and English pronoun capitalization.
    """
    if not text:
        return ""

    text = " ".join(text.strip().split())
    if not text:
        return ""

    # If the text is all-uppercase (standard output from Zipformer ASR)
    if text.isupper():
        # Split into sentence segments
        parts = re.split(r"([.!?]\s*)", text.lower())
        cased = ""
        for p in parts:
            if p:
                cased += p[0].upper() + p[1:] if len(p) > 1 else p.upper()

        # Fix pronoun 'i', contractions 'i'm', 'i'll', 'i'd', 'i've'
        cased = re.sub(r"\bi\b", "I", cased)
        cased = re.sub(r"\bi'([a-z]+)\b", r"I'\1", cased)
        cased = re.sub(r"\bjarvis\b", "Jarvis", cased, flags=re.IGNORECASE)
        return cased

    return text


class StreamingASREngine:
    """
    High-performance real-time streaming Speech-to-Text engine.
    Ingests continuous PCM audio frames and yields real-time partial words
    and automatic endpoint events.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        num_threads: int = 2,
        sample_rate: int = 16000,
        rule1_silence: float = 2.4,
        rule2_silence: float = 0.85,
        rule3_max_utterance: float = 30.0,
    ):
        self.model_name = model_name
        self.num_threads = num_threads
        self.sample_rate = sample_rate
        self.rule1_silence = rule1_silence
        self.rule2_silence = rule2_silence
        self.rule3_max_utterance = rule3_max_utterance

        self.recognizer: Optional[sherpa_onnx.OnlineRecognizer] = None
        self.stream: Optional[sherpa_onnx.OnlineStream] = None
        self._is_loaded = False
        self.last_partial_text = ""

    def _resolve_model_files(self) -> Tuple[str, str, str, str]:
        """Locate model files on disk or download them if missing."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(base_dir, "models")
        primary_dir = os.path.join(models_dir, self.model_name)
        fallback_dir = os.path.join(models_dir, FALLBACK_MODEL_NAME)

        chosen_dir = None
        is_fallback = False

        # 1. Check primary model directory
        if os.path.isdir(primary_dir):
            chosen_dir = primary_dir
        elif os.path.isdir(fallback_dir):
            chosen_dir = fallback_dir
            is_fallback = True
        else:
            # Download model from Hugging Face
            try:
                from huggingface_hub import hf_hub_download
                print(f"[StreamingASR] Downloading streaming Zipformer model ({self.model_name})...", flush=True)
                os.makedirs(primary_dir, exist_ok=True)
                repo_id = f"csukuangfj/{self.model_name}"
                files = [
                    "tokens.txt",
                    "encoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "decoder-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                    "joiner-epoch-99-avg-1-chunk-16-left-128.int8.onnx",
                ]
                for f in files:
                    hf_hub_download(repo_id=repo_id, filename=f, local_dir=primary_dir)
                chosen_dir = primary_dir
            except Exception as e:
                print(f"[StreamingASR] Error downloading {self.model_name}: {e}. Falling back to 20M model...", flush=True)
                os.makedirs(fallback_dir, exist_ok=True)
                from huggingface_hub import hf_hub_download
                repo_id = f"csukuangfj/{FALLBACK_MODEL_NAME}"
                files = [
                    "tokens.txt",
                    "encoder-epoch-99-avg-1.int8.onnx",
                    "decoder-epoch-99-avg-1.int8.onnx",
                    "joiner-epoch-99-avg-1.int8.onnx",
                ]
                for f in files:
                    hf_hub_download(repo_id=repo_id, filename=f, local_dir=fallback_dir)
                chosen_dir = fallback_dir
                is_fallback = True

        tokens = os.path.join(chosen_dir, "tokens.txt")

        # Discover encoder/decoder/joiner onnx files in directory
        all_files = os.listdir(chosen_dir)
        # Prefer int8 quantized models
        encoders = [f for f in all_files if f.startswith("encoder") and f.endswith(".onnx")]
        decoders = [f for f in all_files if f.startswith("decoder") and f.endswith(".onnx")]
        joiners = [f for f in all_files if f.startswith("joiner") and f.endswith(".onnx")]

        encoder_f = next((f for f in encoders if "int8" in f), encoders[0] if encoders else "")
        decoder_f = next((f for f in decoders if "int8" in f), decoders[0] if decoders else "")
        joiner_f = next((f for f in joiners if "int8" in f), joiners[0] if joiners else "")

        encoder = os.path.join(chosen_dir, encoder_f)
        decoder = os.path.join(chosen_dir, decoder_f)
        joiner = os.path.join(chosen_dir, joiner_f)

        if not (os.path.exists(tokens) and os.path.exists(encoder) and os.path.exists(decoder) and os.path.exists(joiner)):
            raise FileNotFoundError(f"Missing ONNX model files in {chosen_dir}")

        return tokens, encoder, decoder, joiner

    def load(self):
        """Initialize the ONNX streaming transducer recognizer."""
        if self._is_loaded and self.recognizer is not None:
            return

        if not HAS_SHERPA_ONNX:
            raise ImportError(
                "sherpa-onnx is not installed. Run 'pip install sherpa-onnx' to enable streaming ASR."
            )

        tokens, encoder, decoder, joiner = self._resolve_model_files()

        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=self.num_threads,
            sample_rate=self.sample_rate,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=self.rule1_silence,
            rule2_min_trailing_silence=self.rule2_silence,
            rule3_min_utterance_length=self.rule3_max_utterance,
        )

        self.stream = self.recognizer.create_stream()
        self._is_loaded = True
        self.last_partial_text = ""

    def reset_stream(self):
        """Reset the current stream to prepare for a new utterance."""
        self.last_partial_text = ""
        if self.recognizer is not None and self.stream is not None:
            self.recognizer.reset(self.stream)

    def accept_waveform(self, pcm_chunk: bytes | np.ndarray) -> str:
        """
        Accept a raw 16-bit PCM chunk or float32 numpy array and run streaming decoding.
        Returns the updated partial transcription string if changed, else empty string.
        """
        if not self._is_loaded or self.recognizer is None or self.stream is None:
            self.load()

        if isinstance(pcm_chunk, bytes):
            audio_f32 = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32) / 32768.0
        elif isinstance(pcm_chunk, np.ndarray):
            if pcm_chunk.dtype == np.int16:
                audio_f32 = pcm_chunk.astype(np.float32) / 32768.0
            else:
                audio_f32 = pcm_chunk.astype(np.float32)
        else:
            return ""

        self.stream.accept_waveform(self.sample_rate, audio_f32)

        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

        raw_partial = self.recognizer.get_result(self.stream).strip()
        formatted_partial = normalize_asr_text(raw_partial)

        if formatted_partial != self.last_partial_text:
            self.last_partial_text = formatted_partial
            return formatted_partial

        return ""

    def get_partial_text(self) -> str:
        """Get the current accumulated partial transcription."""
        if not self._is_loaded or self.recognizer is None or self.stream is None:
            return ""
        raw_partial = self.recognizer.get_result(self.stream).strip()
        return normalize_asr_text(raw_partial)

    def is_endpoint(self) -> bool:
        """Return True if speech pause / endpointing silence threshold has been reached."""
        if not self._is_loaded or self.recognizer is None or self.stream is None:
            return False
        return self.recognizer.is_endpoint(self.stream)

    def get_final_text(self, reset: bool = True) -> str:
        """
        Retrieve finalized transcription string and optionally reset the stream
        for the next round of user speech.
        """
        if not self._is_loaded or self.recognizer is None or self.stream is None:
            return ""

        raw_result = self.recognizer.get_result(self.stream).strip()
        final_text = normalize_asr_text(raw_result)

        if reset:
            self.reset_stream()

        return final_text
