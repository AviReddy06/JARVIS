"""
Ollama Local LLM Engine for JARVIS Voice Assistant.
Features:
- Warm, friendly, down-to-earth conversational human persona.
- Dual-model Shadow Reasoner pipeline with intent routing:
  - DeepSeek-R1 (7b) for coding, mathematics, and complex logic with immediate stall phrase emission.
  - Llama 3.2 (3b) for conversational inquiries and rapid responsiveness.
- Background worker threading for heavy DeepSeek inference to guarantee immediate TTS triggering.
- Real-time streaming and regex post-filtering to strip <think>...</think> reasoning tags.
"""

import json
import logging
import queue
import re
import threading
from typing import Any, Callable, Dict, Generator, List, Optional
import requests

from config import DEFAULT_NUM_CTX, DEFAULT_OLLAMA_HOST, DEFAULT_OLLAMA_MODEL

logger = logging.getLogger(__name__)

# =============================================================================
# Step 1: Conversational Human Persona System Prompt
# =============================================================================
SYSTEM_PROMPT = (
    "You are Jarvis, a warm, friendly, and down-to-earth human companion and personal assistant. "
    "Speak naturally, casually, and directly as a helpful friend in conversational spoken English. "
    "You are Jarvis, so always identify as Jarvis when asked who you are. "
    "Do not use markdown formatting such as asterisks, bold text, italics, or code blocks. "
    "Never use bullet points, numbered lists, or outlines. "
    "Never use robotic AI filler phrases such as 'As an AI', 'As a language model', 'I can help with that', "
    "'Sure thing!', or 'Certainly!'. Keep your responses clear, natural, and concise so they sound great when read aloud."
)

# Dual-Model Constants
MODEL_CONVERSATIONAL = "llama3.2:3b"
MODEL_REASONER = "deepseek-r1:7b"
DEFAULT_STALL_PHRASE = "Let me think about that for a second... "

# Regex pattern for stripping <think>...</think> tags and contents
THINK_REGEX = re.compile(r"<think>[\s\S]*?</think>", flags=re.DOTALL)
CLEAN_THINK_TAGS = re.compile(r"</?think>")

# Intent routing regex and keywords
MATH_REGEX = re.compile(
    r"(\d+\s*[\+\-\*\/\^%]\s*\d+)|(\b\d*[a-zA-Z]\s*[\+\-\*\/]\s*\d+\s*=\s*\d+\b)"
)

CODING_KEYWORDS = {
    "code", "coding", "program", "programming", "python", "javascript", "typescript",
    "java", "c++", "c#", "rust", "golang", "ruby", "php", "swift", "kotlin",
    "html", "css", "sql", "bash", "powershell", "shell", "script", "scripting",
    "function", "functions", "class", "classes", "method", "methods", "variable", "variables",
    "loop", "loops", "array", "arrays", "dictionary", "dictionaries", "list", "lists",
    "algorithm", "algorithms", "debug", "debugging", "debugger", "refactor", "refactoring",
    "compile", "compiler", "compiling", "syntax", "git", "github", "api", "apis",
    "endpoint", "endpoints", "regex", "json", "yaml", "xml", "async", "await",
    "multithreading", "threading", "developer", "development", "bug", "bugs",
    "exception", "exceptions", "traceback", "stack trace", "stacktrace", "frontend", "backend",
    "database", "databases", "query", "queries", "recursion", "recursive", "binary search",
    "data structure", "data structures", "object-oriented", "oop", "decorator", "lambda",
    "pointer", "pointers", "memory leak", "segfault", "segmentation fault", "null pointer",
    "unit test", "pytest", "unittest"
}

MATH_KEYWORDS = {
    "math", "mathematics", "mathematical", "calculate", "calculation", "calculating", "calculator",
    "arithmetic", "equation", "equations", "solve", "solving", "algebra", "algebraic",
    "calculus", "geometry", "geometric", "trigonometry", "trig", "derivative", "derivatives",
    "integral", "integrals", "integration", "differentiate", "differentiation",
    "multiply", "multiplication", "divide", "division", "subtract", "subtraction",
    "square root", "sqrt", "cube root", "percentage", "percent", "fraction", "fractions",
    "logarithm", "logarithms", "log", "ln", "prime", "prime number", "prime numbers",
    "matrix", "matrices", "probability", "statistics", "statistical", "median", "mean",
    "standard deviation", "variance", "formula", "formulas", "theorem", "theorems",
    "pythagorean", "factorial", "factorials", "exponent", "exponents", "exponential",
    "polynomial", "polynomials", "quadratic", "hypotenuse", "sine", "cosine", "tangent"
}

LOGIC_KEYWORDS = {
    "logic", "logical", "reasoning", "reason", "deduce", "deduction", "deductive",
    "riddle", "riddles", "puzzle", "puzzles", "paradox", "paradoxes",
    "brainteaser", "brainteasers", "step by step", "step-by-step", "think through",
    "analyze", "analysis", "analytical", "syllogism", "premise", "premises",
    "fallacy", "fallacies", "implication", "infer", "inference", "trolley problem",
    "game theory", "formal logic"
}

ALL_REASONING_KEYWORDS = CODING_KEYWORDS | MATH_KEYWORDS | LOGIC_KEYWORDS

REASONING_PHRASES = (
    "step by step",
    "step-by-step",
    "think through",
    "square root",
    "cube root",
    "prime number",
    "data structure",
    "game theory",
    "trolley problem",
)


def route_intent(prompt: str) -> str:
    """
    Route prompt to deepseek-r1:7b for coding, mathematics, or complex logic.
    For all other conversational queries, route to llama3.2:3b.
    """
    if not prompt:
        return MODEL_CONVERSATIONAL

    lower = prompt.lower().strip()
    if MATH_REGEX.search(lower):
        return MODEL_REASONER

    words = set(re.findall(r"\b[a-z0-9_\+\-\#\.]+\b", lower))
    if words & ALL_REASONING_KEYWORDS:
        return MODEL_REASONER

    for phrase in REASONING_PHRASES:
        if phrase in lower:
            return MODEL_REASONER

    return MODEL_CONVERSATIONAL


class StreamThinkFilter:
    """
    Filters tokens in real-time to suppress <think>...</think> reasoning blocks
    from being emitted to the TTS engine or user interface.
    """
    def __init__(self):
        self.in_think = False
        self.buffer = ""

    def process(self, token: str) -> str:
        self.buffer += token
        output = ""

        while self.buffer:
            if not self.in_think:
                idx = self.buffer.find("<think>")
                if idx != -1:
                    output += self.buffer[:idx]
                    self.buffer = self.buffer[idx + len("<think>"):]
                    self.in_think = True
                else:
                    longest_suffix_len = 0
                    for i in range(min(len("<think>") - 1, len(self.buffer)), 0, -1):
                        if "<think>".startswith(self.buffer[-i:]):
                            longest_suffix_len = i
                            break
                    if longest_suffix_len > 0:
                        output += self.buffer[:-longest_suffix_len]
                        self.buffer = self.buffer[-longest_suffix_len:]
                    else:
                        output += self.buffer
                        self.buffer = ""
                    break
            else:
                idx = self.buffer.find("</think>")
                if idx != -1:
                    self.buffer = self.buffer[idx + len("</think>"):].lstrip("\n ")
                    self.in_think = False
                else:
                    longest_suffix_len = 0
                    for i in range(min(len("</think>") - 1, len(self.buffer)), 0, -1):
                        if "</think>".startswith(self.buffer[-i:]):
                            longest_suffix_len = i
                            break
                    if longest_suffix_len > 0:
                        self.buffer = self.buffer[-longest_suffix_len:]
                    else:
                        self.buffer = ""
                    break

        return output

    def flush(self) -> str:
        if not self.in_think and self.buffer:
            res = self.buffer
            self.buffer = ""
            return res
        return ""


class OllamaEngine:
    def __init__(self, host: str = DEFAULT_OLLAMA_HOST, num_ctx: int = DEFAULT_NUM_CTX):
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.is_generating = False
        self._cancel_flag = threading.Event()
        self._current_request = None

    def check_connection(self) -> bool:
        """Check if Ollama server is reachable."""
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=2.0)
            return res.status_code == 200
        except Exception:
            return False

    def get_available_models(self) -> List[str]:
        """Fetch list of pulled models from Ollama."""
        try:
            res = requests.get(f"{self.host}/api/tags", timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name") for m in data.get("models", []) if m.get("name")]
                return models if models else [DEFAULT_OLLAMA_MODEL]
        except Exception as e:
            logger.warning(f"Failed to fetch models from Ollama: {e}")
        return [DEFAULT_OLLAMA_MODEL]

    def stop_generation(self):
        """Signal to abort any currently streaming generation."""
        self._cancel_flag.set()
        self.is_generating = False

    def route_intent(self, prompt: str) -> str:
        """Expose route_intent on the engine instance."""
        return route_intent(prompt)

    def stream_chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        num_ctx: Optional[int] = None,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_complete: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> Generator[str, None, None]:
        """
        Stream chat completion tokens with dual-model Shadow Reasoner routing.
        - Warm human companion persona with SYSTEM_PROMPT prepended.
        - Routing: deepseek-r1:7b for code/math/logic, llama3.2:3b for conversational queries.
        - DeepSeek path immediately yields stall phrase and runs inference on background thread.
        - Real-time and regex filtering strips <think>...</think> tags before TTS.
        """
        self.is_generating = True
        self._cancel_flag.clear()

        # Step 2: Ensure SYSTEM_PROMPT is always inserted first as a "system" role message
        # before the conversation history
        sanitized_history = [m for m in messages if m.get("role") != "system"]
        formatted_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + sanitized_history

        # Determine user prompt for intent routing
        user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                user_prompt = m.get("content", "")
                break

        # Step 3: Implement intent router to select target model
        if model and model not in (DEFAULT_OLLAMA_MODEL, MODEL_CONVERSATIONAL, MODEL_REASONER, "llama3.2:latest"):
            target_model = model
        else:
            target_model = route_intent(user_prompt)

        # Step 2: Verify temperature remains 0.7, set expanded context window
        chosen_ctx = num_ctx or self.num_ctx
        endpoint = f"{self.host}/api/chat"
        payload = {
            "model": target_model,
            "messages": formatted_messages,
            "stream": True,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "num_ctx": chosen_ctx,
            },
        }

        metrics = {
            "model": target_model,
            "total_tokens": 0,
            "eval_duration_sec": 0.0,
            "text": "",
        }

        try:
            if target_model == MODEL_REASONER:
                # Step 4: Immediately yield conversational stall phrase
                stall_phrase = DEFAULT_STALL_PHRASE
                if on_chunk:
                    on_chunk(stall_phrase)
                yield stall_phrase

                token_queue: queue.Queue = queue.Queue()

                # Step 4: Process actual DeepSeek inference on a background thread
                def _deepseek_worker():
                    try:
                        with requests.post(
                            endpoint,
                            json=payload,
                            stream=True,
                            timeout=90.0,
                        ) as response:
                            if response.status_code != 200:
                                err_msg = f"Ollama HTTP error {response.status_code}: {response.text}"
                                token_queue.put(("error", err_msg))
                                return

                            for line in response.iter_lines(decode_unicode=True):
                                if self._cancel_flag.is_set():
                                    break
                                if not line:
                                    continue
                                try:
                                    chunk = json.loads(line)
                                    msg = chunk.get("message", {})
                                    content = msg.get("content", "")
                                    if content:
                                        token_queue.put(("chunk", content))
                                    if chunk.get("done", False):
                                        token_queue.put(("done", chunk))
                                except json.JSONDecodeError:
                                    continue
                    except requests.exceptions.ConnectionError:
                        token_queue.put(("error", "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434."))
                    except Exception as ex:
                        token_queue.put(("error", str(ex)))
                    finally:
                        token_queue.put(("eof", None))

                worker_thread = threading.Thread(target=_deepseek_worker, daemon=True)
                worker_thread.start()

                # Step 5: Filter <think>...</think> tags before passing to TTS
                think_filter = StreamThinkFilter()
                raw_response_tokens: List[str] = []
                clean_response_tokens: List[str] = []

                while True:
                    if self._cancel_flag.is_set():
                        break
                    try:
                        msg_type, data = token_queue.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    if msg_type == "eof":
                        break
                    elif msg_type == "error":
                        if on_error and not self._cancel_flag.is_set():
                            on_error(data)
                        return
                    elif msg_type == "chunk":
                        raw_response_tokens.append(data)
                        clean_chunk = think_filter.process(data)
                        if clean_chunk:
                            clean_response_tokens.append(clean_chunk)
                            if on_chunk:
                                on_chunk(clean_chunk)
                            yield clean_chunk
                    elif msg_type == "done":
                        eval_dur = data.get("eval_duration", 0) / 1e9
                        metrics["eval_duration_sec"] = round(eval_dur, 2)
                        metrics["total_tokens"] = data.get("eval_count", len(raw_response_tokens))

                # Flush any leftover non-think buffer
                flushed = think_filter.flush()
                if flushed:
                    clean_response_tokens.append(flushed)
                    if on_chunk:
                        on_chunk(flushed)
                    yield flushed

                # Step 5: Regex filter to strip all <think>...</think> tags before final clean text
                full_raw = "".join(raw_response_tokens)
                final_clean = THINK_REGEX.sub("", full_raw)
                final_clean = CLEAN_THINK_TAGS.sub("", final_clean).strip()
                if not final_clean and clean_response_tokens:
                    final_clean = "".join(clean_response_tokens).strip()

                metrics["text"] = final_clean
                if on_complete and not self._cancel_flag.is_set():
                    on_complete(metrics)

            else:
                # Conversational path (llama3.2:3b)
                full_response = []
                with requests.post(
                    endpoint,
                    json=payload,
                    stream=True,
                    timeout=60.0,
                ) as response:
                    if response.status_code != 200:
                        err_msg = f"Ollama HTTP error {response.status_code}: {response.text}"
                        if on_error:
                            on_error(err_msg)
                        return

                    for line in response.iter_lines(decode_unicode=True):
                        if self._cancel_flag.is_set():
                            break
                        if not line:
                            continue
                        try:
                            chunk = json.loads(line)
                            msg = chunk.get("message", {})
                            token = msg.get("content", "")
                            if token:
                                full_response.append(token)
                                if on_chunk:
                                    on_chunk(token)
                                yield token

                            if chunk.get("done", False):
                                eval_dur = chunk.get("eval_duration", 0) / 1e9
                                metrics["eval_duration_sec"] = round(eval_dur, 2)
                                metrics["total_tokens"] = chunk.get("eval_count", len(full_response))
                        except json.JSONDecodeError:
                            continue

                metrics["text"] = "".join(full_response)
                if on_complete and not self._cancel_flag.is_set():
                    on_complete(metrics)

        except requests.exceptions.ConnectionError:
            err = "Cannot connect to Ollama. Make sure Ollama is running on localhost:11434."
            if on_error:
                on_error(err)
        except Exception as e:
            if not self._cancel_flag.is_set() and on_error:
                on_error(str(e))
        finally:
            self.is_generating = False
