"""
Automated Verification Suite for JARVIS Real-Time Streaming ASR Pipeline.
Tests:
1. StreamingASREngine initialization, loading, chunked waveform ingestion (50ms - 100ms frames).
2. Word-by-word partial emissions and endpoint detection.
3. Text normalization (casing, punctuation, pronouns).
4. AudioListenerThread signals (wake_signal, update_text_signal, sleep_signal, partial_text_signal).
5. JarvisHUD live partial text handling and UI positioning.
6. Integration with TTSEngine, WakeWordEngine, and OllamaEngine.
"""

import os
import sys
import unittest
import numpy as np
import soundfile as sf
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from config import (
    DEFAULT_STREAMING_MODEL,
    SAMPLE_RATE,
    USE_STREAMING_ASR,
)
from streaming_asr import StreamingASREngine, normalize_asr_text
from voice_engine import VoiceEngine
from jarvis_ui import AudioListenerThread, JarvisHUD
from ollama_engine import OllamaEngine
from wakeword_engine import WakeWordEngine


class TestStreamingASREngine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = StreamingASREngine()
        cls.engine.load()

        # Path to test wave file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cls.test_wav = os.path.join(
            base_dir,
            "models",
            "sherpa-onnx-streaming-zipformer-en-20M-2023-02-17",
            "test_wavs",
            "0.wav"
        )
        if not os.path.exists(cls.test_wav):
            cls.test_wav = None

    def test_engine_loaded(self):
        """Verify the Zipformer streaming engine loads and creates stream."""
        self.assertTrue(self.engine._is_loaded, "Engine failed to load")
        self.assertIsNotNone(self.engine.recognizer, "Recognizer should not be None")
        self.assertIsNotNone(self.engine.stream, "Stream should not be None")

    def test_text_normalization(self):
        """Verify capitalization and grammar normalization."""
        raw = "HELLO WORLD THIS IS JARVIS"
        norm = normalize_asr_text(raw)
        self.assertEqual(norm, "Hello world this is Jarvis")

        raw_i = "I THINK I'M READY FOR THE TEST"
        norm_i = normalize_asr_text(raw_i)
        self.assertIn("I", norm_i)
        self.assertIn("I'm", norm_i)

    def test_chunked_streaming_transcription(self):
        """Verify 50ms - 80ms chunks yield word-by-word partial results."""
        if not self.test_wav:
            self.skipTest("Test wav file not available")

        samples, sr = sf.read(self.test_wav, dtype="float32")
        self.assertEqual(sr, 16000, "Sample rate must be 16kHz")

        self.engine.reset_stream()
        chunk_size = int(0.08 * sr)  # 80ms chunk
        partials = []

        for i in range(0, len(samples), chunk_size):
            chunk = samples[i:i + chunk_size]
            updated = self.engine.accept_waveform(chunk)
            if updated:
                partials.append(updated)

        self.assertGreater(len(partials), 3, "Streaming should emit multiple word-by-word partial updates")
        final_text = self.engine.get_final_text(reset=True)
        self.assertTrue(len(final_text) > 0, "Final text should not be empty")
        self.assertIn("yellow lamps", final_text.lower(), "Transcription should match audio contents")

    def test_endpoint_detection_with_trailing_silence(self):
        """Verify endpointing accurately triggers upon silence after speech."""
        if not self.test_wav:
            self.skipTest("Test wav file not available")

        samples, sr = sf.read(self.test_wav, dtype="float32")
        # 3 seconds speech + 2.0 seconds silence
        speech = samples[:int(3.0 * sr)]
        silence = np.zeros(int(2.0 * sr), dtype=np.float32)
        combined = np.concatenate([speech, silence])

        self.engine.reset_stream()
        chunk_size = int(0.08 * sr)
        endpoint_hit = False

        for i in range(0, len(combined), chunk_size):
            chunk = combined[i:i + chunk_size]
            self.engine.accept_waveform(chunk)
            if self.engine.is_endpoint():
                endpoint_hit = True
                final_text = self.engine.get_final_text(reset=True)
                self.assertTrue(len(final_text) > 0, "Final text should be captured at endpoint")
                break

        self.assertTrue(endpoint_hit, "Endpoint should be detected after trailing silence")


class TestVoiceEngineStreamingIntegration(unittest.TestCase):
    def setUp(self):
        self.voice_engine = VoiceEngine()

    def tearDown(self):
        self.voice_engine.cleanup()

    def test_voice_engine_has_streaming_asr(self):
        """Verify VoiceEngine initializes with streaming ASR engine."""
        self.assertTrue(hasattr(self.voice_engine, "streaming_engine"))
        self.assertTrue(self.voice_engine.use_streaming)

    def test_streaming_asr_load(self):
        """Verify VoiceEngine can load streaming ASR in memory."""
        self.voice_engine.load_streaming_asr()
        self.assertTrue(self.voice_engine.streaming_engine._is_loaded)


class TestPyQtHUDStreamingSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(sys.argv)

    def setUp(self):
        self.hud = JarvisHUD()
        self.listener = AudioListenerThread()

    def tearDown(self):
        self.hud.close()
        self.listener.stop()

    def test_listener_thread_signals(self):
        """Verify AudioListenerThread has all 4 required PyQt signals including partial_text_signal."""
        self.assertTrue(hasattr(self.listener, "wake_signal"), "Missing wake_signal")
        self.assertTrue(hasattr(self.listener, "update_text_signal"), "Missing update_text_signal")
        self.assertTrue(hasattr(self.listener, "sleep_signal"), "Missing sleep_signal")
        self.assertTrue(hasattr(self.listener, "partial_text_signal"), "Missing partial_text_signal")

    def test_hud_handles_word_by_word_partial_text(self):
        """Verify HUD smoothly handles real-time partial text updates without error."""
        self.hud.on_wake()
        self.assertTrue(self.hud.isVisible())

        # Simulate real-time word-by-word streaming updates
        words = ["What", "What is", "What is the", "What is the capital", "What is the capital of", "What is the capital of France?"]
        for w in words:
            self.hud.on_update_text(f"🎙️ <b>Listening...</b><br><span style='color:#FFFFFF;font-size:16px;'><b>You: </b>{w} ●</span>")
            self.hud.on_partial_text(w)
            self.assertIn(w, self.hud.label.text())

        # Verify geometry updated smoothly
        geom = self.hud.geometry()
        self.assertGreater(geom.width(), 400)


class TestEngineCompatibility(unittest.TestCase):
    def test_ollama_engine_handoff_signature(self):
        """Verify ollama_engine can immediately receive final_result query string."""
        engine = OllamaEngine()
        target_model = engine.route_intent("What is the speed of light?")
        self.assertIn("llama3.2", target_model.lower())

    def test_wakeword_engine_compatibility(self):
        """Verify wakeword_engine can be loaded and initialized alongside streaming ASR."""
        ww = WakeWordEngine()
        self.assertEqual(ww.model_name_key, "hey_jarvis_v0.1")


if __name__ == "__main__":
    unittest.main()
