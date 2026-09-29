# 🎙️ JARVIS AI Voice Assistant (v2.0)

A complete local AI voice assistant featuring:
1. **Wake Word Detection**: **openWakeWord** listens locally for **"Hey Jarvis"**.
2. **Speech-to-Text**: **sherpa-onnx** streaming Zipformer transducer transcribes speech in real time with word-by-word UI feedback and automatic silence/endpoint detection.
3. **Audio Capture**: **sounddevice** continuously streams low-latency microphone frames (80ms) directly into sherpa-onnx with zero batch buffering.
4. **Reasoning & Chat**: **Ollama (`llama3.2`)** processes your requests and streams text answers.
5. **Voice Synthesis**: **Kokoro TTS** speaks the answers aloud with a British gentleman voice (`bm_george`) via **sounddevice**!
6. **Multimodal Screen Vision**: **Moondream** inspects your active screen in real time to tell you what's happening or answer questions about your windows and code!

---

## ⚡ How to Interact

1. **Say "Hey Jarvis"**
   - JARVIS immediately activates and responds: *"Yes sir?"*
2. **Speak your question**
   - General AI: *"What's the time and what's on the agenda today?"*
   - Screen Vision: *"What's on my screen right now?"* or *"Look at my screen and summarize this."*
3. **Real-Time Word-by-Word Feedback**
   - As you speak, each word appears immediately on screen.
   - Upon natural pause/silence, JARVIS instantly begins answering with zero batch latency!
4. **Listen to JARVIS answer**
   - The answer streams on the HUD and is spoken aloud in high-definition natural voice!
5. **Follow-up conversation**
   - JARVIS enters an active conversation window so you can ask follow-up questions without repeating the wake word.
   - If you want him to sleep immediately, just say *"Go to sleep"*, *"That's all"*, or *"Standby"*.

---

## 🚀 How to Run

### 1. 🌟 Graphical Background HUD Service (PyQt6 - Modern)
Run silently in the background with zero terminal windows:
```bash
pythonw jarvis_ui.pyw
```
*(or double-click **`jarvis_ui.pyw`** or **`run_hud.bat`**)*

- **Completely invisible on startup**
- **HUD appears on "Hey Jarvis"** centered at the bottom of your primary screen
- **Real-time word-by-word streaming speech feedback**
- **Displays transcriptions and real-time streaming AI answers**
- **Screen Vision**: Ask *"What's happening on my screen?"* to analyze your active desktop in real time
- **Multi-turn conversational flow** without repeating wake word
- **Smoothly fades out and disappears** when finished
- **System tray icon** with quick test controls, screen vision, and clean shutdown

### 2. 🖥️ Terminal Edition
```bash
python app.py
```
*(or double-click **`run.bat`**)*

---

## 📦 AI Stack

| Component | Library / Engine | Model |
| --- | --- | --- |
| **Wake Word** | `openwakeword` | `hey_jarvis_v0.1.onnx` |
| **Audio Ingestion** | `sounddevice` | Continuous stream (`float32`, 80ms frames) |
| **Speech-to-Text** | `sherpa-onnx` | Streaming Zipformer transducer (Int8 ONNX) |
| **Reasoning / LLM**| `Ollama` | `llama3.2` |
| **Multimodal Vision**| `Ollama` | `moondream` (GPU accelerated) |
| **Voice Synthesis**| `kokoro-onnx` + `sounddevice` | Kokoro-82M (`bm_george`) |
