"""
JARVIS Voice Assistant - Terminal Edition (v2.0)
Integrated with:
- openWakeWord: "Hey Jarvis" local wake word detection
- sherpa-onnx: Real-time streaming Zipformer transducer ASR
- Ollama (llama3.2): Local LLM reasoning with real-time text streaming
- Kokoro TTS: High-fidelity British voice synthesis via sounddevice
"""

import datetime
import os
import queue
import sys
import threading
import time
from typing import Optional

from config import (
    APP_NAME,
    APP_VERSION,
    DEFAULT_MAX_CHAT_HISTORY,
    DEFAULT_NUM_CTX,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_PAUSE_THRESHOLD,
    DEFAULT_STREAMING_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_VOICE,
    DEFAULT_VOICE_SPEED,
    USE_STREAMING_ASR,
    WAKE_WORD_REQUIRED,
)
from ollama_engine import OllamaEngine
from tts_engine import TTSEngine
from voice_engine import VoiceEngine

# Ensure Windows terminal prints UTF-8 and ANSI colors properly
sys.stdout.reconfigure(encoding="utf-8")
os.system("")  # Enable ANSI escape sequences on Windows console

# ANSI Colors
CLR_RESET = "\033[0m"
CLR_BOLD = "\033[1m"
CLR_DIM = "\033[2m"
CLR_CYAN = "\033[96m"
CLR_BOLD_CYAN = "\033[1;96m"
CLR_GREEN = "\033[92m"
CLR_BOLD_GREEN = "\033[1;92m"
CLR_YELLOW = "\033[93m"
CLR_RED = "\033[91m"
CLR_PURPLE = "\033[95m"
CLR_BOLD_PURPLE = "\033[1;95m"


class JarvisApp:
    def __init__(self, wake_word_required: Optional[bool] = None):
        self.voice_engine = VoiceEngine()
        self.ollama_engine = OllamaEngine(num_ctx=DEFAULT_NUM_CTX)
        self.tts_engine = TTSEngine(voice=DEFAULT_VOICE, speed=DEFAULT_VOICE_SPEED)

        self.model_name = DEFAULT_OLLAMA_MODEL
        self.chat_history = []
        self.work_queue = queue.Queue()
        self.is_running = True

        # State tracking
        self.wake_word_required = WAKE_WORD_REQUIRED if wake_word_required is None else wake_word_required
        self.awake = not self.wake_word_required
        self.last_interaction_time = time.time()
        self.awake_timeout_seconds = 15.0  # Go to sleep after 15s of silence

    def print_banner(self):
        mode_str = "Wake Word ('Hey Jarvis')" if self.wake_word_required else "Continuous Live Mode"
        banner = f"""
{CLR_BOLD_PURPLE}╔══════════════════════════════════════════════════════════════════╗
║                   🎙️   JARVIS AI VOICE ASSISTANT                  ║
║      Mode: {mode_str:<20} • STT (sherpa-onnx) • TTS (Kokoro)  ║
║                      LLM: Ollama ({self.model_name})                     ║
╚══════════════════════════════════════════════════════════════════╝{CLR_RESET}
"""
        print(banner)

    def initialize_services(self):
        """Preload all AI models."""
        print(f"{CLR_DIM}[1/4] Checking Ollama connection...{CLR_RESET}", end="", flush=True)
        if self.ollama_engine.check_connection():
            models = self.ollama_engine.get_available_models()
            matched = next((m for m in models if "llama3.2" in m.lower()), models[0] if models else DEFAULT_OLLAMA_MODEL)
            self.model_name = matched
            print(f"\r{CLR_GREEN}✓ [1/4] Ollama Online ({self.model_name}){CLR_RESET}               ")
        else:
            print(f"\r{CLR_RED}✗ [1/4] Ollama Offline. (Run 'ollama serve' in background){CLR_RESET}")

        print(f"{CLR_DIM}[2/4] Loading real-time streaming Zipformer ASR model...{CLR_RESET}", end="", flush=True)
        try:
            self.voice_engine.load_streaming_asr()
            print(f"\r{CLR_GREEN}✓ [2/4] Streaming Zipformer ASR Ready in memory{CLR_RESET}               ")
        except Exception as e:
            print(f"\r{CLR_RED}✗ [2/4] Streaming ASR error: {e}{CLR_RESET}")

        print(f"{CLR_DIM}[3/4] Initializing Kokoro Voice TTS ({DEFAULT_VOICE})...{CLR_RESET}", end="", flush=True)
        try:
            self.tts_engine.load()
            print(f"\r{CLR_GREEN}✓ [3/4] Kokoro TTS Ready ({DEFAULT_VOICE}){CLR_RESET}                ")
        except Exception as e:
            print(f"\r{CLR_RED}✗ [3/4] Kokoro TTS error: {e}{CLR_RESET}")

        if self.wake_word_required:
            print(f"{CLR_DIM}[4/4] Loading openWakeWord ('Hey Jarvis')...{CLR_RESET}", end="", flush=True)
            try:
                self.voice_engine.load_wakeword_model()
                print(f"\r{CLR_GREEN}✓ [4/4] openWakeWord ('Hey Jarvis') Ready{CLR_RESET}                ")
            except Exception as e:
                print(f"\r{CLR_RED}✗ [4/4] Wake word error: {e}{CLR_RESET}")
        else:
            print(f"{CLR_GREEN}✓ [4/4] Continuous Live Mode (No wake-word needed){CLR_RESET}")

        # Active microphone info
        devices = self.voice_engine.get_input_devices()
        mic_name = devices[0][1] if devices else "Default System Microphone"

        print(f"\n{CLR_BOLD_GREEN}✨ All Systems Online!{CLR_RESET}")
        print(f"{CLR_DIM}• Active Microphone: {CLR_CYAN}{mic_name}{CLR_RESET}")
        if self.wake_word_required:
            print(f"{CLR_DIM}• Say {CLR_BOLD_CYAN}\"Hey Jarvis\"{CLR_RESET}{CLR_DIM} to activate at any time.")
            print(f"{CLR_DIM}• (Tip: run with {CLR_BOLD}'--live'{CLR_DIM} for continuous live mode without wake word){CLR_RESET}")
        else:
            print(f"{CLR_DIM}• Speak naturally into your microphone at any time.")
        print(f"{CLR_DIM}• Press {CLR_BOLD}Ctrl+C{CLR_RESET}{CLR_DIM} to exit safely.{CLR_RESET}")
        print(f"{CLR_DIM}{'─' * 66}{CLR_RESET}\n")

    def run(self):
        """Main event loop."""
        self.print_banner()
        self.initialize_services()

        # Startup greeting
        self.tts_engine.speak("All systems online, sir. Standing by.", blocking=True)

        # Start live audio thread
        self.voice_engine.start_live_mode(
            use_wakeword=self.wake_word_required,
            pause_threshold=DEFAULT_PAUSE_THRESHOLD,
            on_wakeword=self._on_wakeword_triggered,
            on_speech_start=self._on_speech_start,
            on_partial_transcription=self._on_partial_transcription,
            on_endpoint_detected=self._on_endpoint_detected,
            on_error=self._on_audio_error,
        )

        if self.wake_word_required:
            print(f"{CLR_YELLOW}👂 [Standby] Say \"Hey Jarvis\" to wake me up...{CLR_RESET}", end="", flush=True)
        else:
            print(f"{CLR_GREEN}● [Listening] Speak your question at any time...{CLR_RESET}", end="", flush=True)

        try:
            while self.is_running:
                try:
                    event_type, data = self.work_queue.get(timeout=0.1)

                    if event_type == "wakeword_detected":
                        self.awake = True
                        self.last_interaction_time = time.time()
                        print(f"\r{CLR_BOLD_GREEN}⚡ [Activated] \"Hey Jarvis\" detected!{CLR_RESET}                   ")
                        # Jarvis verbal acknowledge
                        self.tts_engine.speak("Yes sir?", blocking=True)
                        print(f"{CLR_GREEN}🎙️  [Listening] What can I do for you, sir?{CLR_RESET}", end="", flush=True)
                        # CRITICAL: Resume live audio recording to hear the user's question!
                        self.voice_engine.resume_live_mode(wait_for_wakeword=False)

                    elif event_type == "speech_started":
                        self.last_interaction_time = time.time()
                        print(f"\r{CLR_YELLOW}🎙️  [Hearing You...] Speak your question (pause when done)...   {CLR_RESET}", end="", flush=True)

                    elif event_type == "partial_transcription":
                        partial = data
                        self.last_interaction_time = time.time()
                        sys.stdout.write(f"\r{CLR_YELLOW}🎙️  You: {partial} ●{CLR_RESET}\033[K")
                        sys.stdout.flush()

                    elif event_type == "endpoint_detected":
                        query_text = data
                        self.last_interaction_time = time.time()
                        print(f"\r{CLR_BOLD_CYAN}🎙️  You: {query_text}{CLR_RESET}\033[K")
                        # Check for sleep commands
                        lower = query_text.lower().strip()
                        if any(cmd in lower for cmd in ["go to sleep", "goodbye", "standby", "stop listening"]):
                            print(f"{CLR_BOLD_PURPLE}🤖 JARVIS: {CLR_RESET}Entering standby mode, sir.")
                            self.tts_engine.speak("Standing by, sir.", blocking=True)
                            self.awake = False
                            print(f"{CLR_YELLOW}👂 [Standby] Say \"Hey Jarvis\" to wake me up...{CLR_RESET}", end="", flush=True)
                            self.voice_engine.resume_live_mode(wait_for_wakeword=True)
                        else:
                            self._generate_and_speak(query_text)

                    elif event_type == "audio_error":
                        err_msg = data
                        print(f"\r{CLR_RED}⚠ Audio notice: {err_msg}{CLR_RESET}")
                        self._reset_listener_prompt()

                except queue.Empty:
                    # Check if awake session has timed out back to standby
                    if self.wake_word_required and self.awake and not self.tts_engine.is_speaking():
                        if time.time() - self.last_interaction_time > self.awake_timeout_seconds:
                            self.awake = False
                            print(f"\n{CLR_DIM}[Standby mode engaged]{CLR_RESET}")
                            print(f"{CLR_YELLOW}👂 [Standby] Say \"Hey Jarvis\" to wake me up...{CLR_RESET}", end="", flush=True)
                            self.voice_engine.resume_live_mode(wait_for_wakeword=True)
                    continue

        except KeyboardInterrupt:
            print(f"\n\n{CLR_BOLD_CYAN}JARVIS powering down. Have a good day, sir! 👋{CLR_RESET}")
        finally:
            self.tts_engine.stop()
            self.voice_engine.cleanup()
            self.ollama_engine.stop_generation()

    def _on_wakeword_triggered(self):
        if getattr(self, "is_responding", False) or self.tts_engine.is_speaking():
            self.is_responding = False
            self.voice_engine.set_barge_in(False)
            self.tts_engine.stop()
            self.ollama_engine.stop_generation()
            print(f"\n{CLR_YELLOW}⚡ [Interrupted by wake word]{CLR_RESET}")

        self.work_queue.put(("wakeword_detected", None))

    def _on_speech_start(self):
        if getattr(self, "is_responding", False) or self.tts_engine.is_speaking():
            self.is_responding = False
            self.voice_engine.set_barge_in(False)
            self.tts_engine.stop()
            self.ollama_engine.stop_generation()
            print(f"\n{CLR_YELLOW}⚡ [Interrupted by your speech! Hearing you...]{CLR_RESET}")

        self.work_queue.put(("speech_started", None))

    def _on_partial_transcription(self, partial_text: str):
        self.work_queue.put(("partial_transcription", partial_text))

    def _on_endpoint_detected(self, final_text: str):
        self.work_queue.put(("endpoint_detected", final_text))

    def _on_audio_error(self, err_msg):
        self.work_queue.put(("audio_error", err_msg))

    def _generate_and_speak(self, prompt: str):
        """Stream answer from Ollama to terminal and speak out loud with Kokoro TTS."""
        self.chat_history.append({"role": "user", "content": prompt})

        # Inject real-time timestamp and date context
        now = datetime.datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")
        live_context = (
            f"{DEFAULT_SYSTEM_PROMPT}\n\n"
            f"[Live System Context]\n"
            f"- Current Local Time: {time_str}\n"
            f"- Current Date: {date_str}\n"
            f"Always use this live context when asked about the time, date, or day."
        )

        messages = [{"role": "system", "content": live_context}]
        messages.extend(self.chat_history[-DEFAULT_MAX_CHAT_HISTORY:])

        print(f"{CLR_BOLD_PURPLE}🤖 JARVIS: {CLR_RESET}", end="", flush=True)

        full_tokens = []
        # Start real-time speech stream
        self.tts_engine.start_stream()
        self.is_responding = True
        self.voice_engine.set_barge_in(True)
        # Enable wake word monitoring during response so user can interrupt at any moment!
        self.voice_engine.resume_live_mode(wait_for_wakeword=True)

        def _on_chunk(token: str):
            if not getattr(self, "is_responding", False):
                return
            full_tokens.append(token)
            sys.stdout.write(token)
            sys.stdout.flush()
            # Feed token in real-time to Kokoro TTS for sentence-level speaking
            self.tts_engine.feed_stream(token)

        def _on_complete(metrics):
            if not getattr(self, "is_responding", False):
                return
            print()  # newline after streamed text
            dur = metrics.get("eval_duration_sec", 0.0)
            tokens = metrics.get("total_tokens", len(full_tokens))
            print(f"{CLR_DIM}[{dur}s • {tokens} tokens]{CLR_RESET}")
            print(f"{CLR_DIM}{'─' * 66}{CLR_RESET}")

            response_text = "".join(full_tokens)
            self.chat_history.append({"role": "assistant", "content": response_text})

            # Finish any remaining sentences and wait for speech to conclude
            self.tts_engine.end_stream(wait=True)
            self.is_responding = False
            self.voice_engine.set_barge_in(False)

            # Update interaction time and resume listening for follow-up questions
            self.last_interaction_time = time.time()
            self._reset_listener_prompt()

        def _on_error(err: str):
            self.is_responding = False
            self.tts_engine.stop()
            print(f"\n{CLR_RED}[Ollama Error: {err}]{CLR_RESET}")
            self._reset_listener_prompt()

        for _ in self.ollama_engine.stream_chat(
            messages=messages,
            model=self.model_name,
            on_chunk=_on_chunk,
            on_complete=_on_complete,
            on_error=_on_error,
        ):
            if not getattr(self, "is_responding", False):
                break

    def _reset_listener_prompt(self):
        time.sleep(0.2)
        if self.wake_word_required and not self.awake:
            print(f"{CLR_YELLOW}👂 [Standby] Say \"Hey Jarvis\" to wake me up...{CLR_RESET}", end="", flush=True)
            self.voice_engine.resume_live_mode(wait_for_wakeword=True)
        else:
            print(f"{CLR_GREEN}● [Listening] I'm listening... Speak at any time.{CLR_RESET}", end="", flush=True)
            self.voice_engine.resume_live_mode(wait_for_wakeword=False)


def main():
    wake_required = None
    args = [a.lower() for a in sys.argv[1:]]
    if "--live" in args or "-l" in args or "--no-wake" in args:
        wake_required = False
    elif "--wake" in args or "-w" in args:
        wake_required = True

    app = JarvisApp(wake_word_required=wake_required)
    app.run()


if __name__ == "__main__":
    main()
