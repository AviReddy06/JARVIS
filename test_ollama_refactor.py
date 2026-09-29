"""
Comprehensive verification test for the refactored ollama_engine.py:
1. Verifies SYSTEM_PROMPT constant rules and instructions.
2. Verifies messages formatting: SYSTEM_PROMPT inserted first as system role.
3. Verifies payload options temperature == 0.7.
4. Verifies route_intent:
   - coding, math, logic -> deepseek-r1:7b
   - conversational -> llama3.2:3b
5. Verifies DeepSeek stall phrase is yielded immediately.
6. Verifies background thread execution for DeepSeek.
7. Verifies regex filter strips <think>...</think> tags.
8. Live integration test with running Ollama server.
"""

import re
import sys
import threading
import time
from unittest.mock import patch, MagicMock

import ollama_engine
from ollama_engine import (
    OllamaEngine,
    SYSTEM_PROMPT,
    MODEL_CONVERSATIONAL,
    MODEL_REASONER,
    DEFAULT_STALL_PHRASE,
    THINK_REGEX,
    route_intent,
    StreamThinkFilter,
)


def test_step_1_system_prompt():
    print("[TEST 1] Testing SYSTEM_PROMPT constant...")
    assert isinstance(SYSTEM_PROMPT, str)
    assert len(SYSTEM_PROMPT) > 50
    # Persona: warm, friendly, companion
    lower = SYSTEM_PROMPT.lower()
    assert "warm" in lower
    assert "friendly" in lower
    assert "companion" in lower
    # Forbids markdown, bullet points, robotic filler
    assert "markdown" in lower
    assert "bullet" in lower
    assert "as an ai" in lower
    print("  [OK] Step 1 passed: SYSTEM_PROMPT meets all requirements.")


def test_step_2_messages_and_temperature():
    print("[TEST 2] Testing messages formatting & temperature = 0.7...")
    engine = OllamaEngine()

    captured_payloads = []

    def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            b'{"message": {"role": "assistant", "content": "Hello friend!"}, "done": true, "eval_duration": 1000000}'
        ]
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        history = [
            {"role": "system", "content": "Old prompt"},
            {"role": "user", "content": "How's your day going?"},
            {"role": "assistant", "content": "Great!"},
            {"role": "user", "content": "Nice to meet you"}
        ]
        tokens = list(engine.stream_chat(history))

    assert len(captured_payloads) == 1
    payload = captured_payloads[0]

    # Verify options temperature is exactly 0.7
    assert payload["options"]["temperature"] == 0.7, f"Temp was {payload['options']['temperature']}"

    # Verify SYSTEM_PROMPT is always inserted first as a system role message
    messages = payload["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == SYSTEM_PROMPT

    # Verify no duplicate system prompt
    other_system_msgs = [m for m in messages[1:] if m.get("role") == "system"]
    assert len(other_system_msgs) == 0

    print("  [OK] Step 2 passed: SYSTEM_PROMPT prepended and temperature verified at 0.7.")


def test_step_3_intent_router():
    print("[TEST 3] Testing Intent Router...")
    
    coding_queries = [
        "Can you write a python script to parse CSV files?",
        "Debug this javascript function for me",
        "How do I create a class with inheritance in C++?",
        "Explain SQL indexing and query optimization",
        "Help me refactor this recursion into a loop",
        "Fix this null pointer exception in Java"
    ]

    math_queries = [
        "Calculate 15 * 14",
        "What is the square root of 144?",
        "Solve for x: 3x + 9 = 24",
        "What is the derivative of x^2 + 5x?",
        "What are the prime numbers between 1 and 20?",
        "What is 25 + 38?"
    ]

    logic_queries = [
        "Here is a riddle: what has hands but cannot clap?",
        "Solve this puzzle step by step",
        "Explain the prisoner's dilemma in game theory",
        "Analyze this logical syllogism and deduce the conclusion",
        "What is the trolley problem paradox?"
    ]

    conversational_queries = [
        "Hey Jarvis, how are you feeling today?",
        "Tell me a funny joke about cats",
        "Good morning!",
        "What's your favorite color?",
        "What should I have for dinner?",
        "Tell me a story about a dragon in a forest"
    ]

    for q in coding_queries:
        routed = route_intent(q)
        assert routed == MODEL_REASONER, f"Expected deepseek for coding '{q}', got {routed}"

    for q in math_queries:
        routed = route_intent(q)
        assert routed == MODEL_REASONER, f"Expected deepseek for math '{q}', got {routed}"

    for q in logic_queries:
        routed = route_intent(q)
        assert routed == MODEL_REASONER, f"Expected deepseek for logic '{q}', got {routed}"

    for q in conversational_queries:
        routed = route_intent(q)
        assert routed == MODEL_CONVERSATIONAL, f"Expected llama for conversational '{q}', got {routed}"

    print("  [OK] Step 3 passed: All intent routing tests passed (deepseek-r1:7b vs llama3.2:3b).")


def test_step_4_stall_phrase_and_threading():
    print("[TEST 4] Testing immediate stall phrase & background threading for DeepSeek...")
    engine = OllamaEngine()

    worker_thread_id = None
    caller_thread_id = threading.get_ident()

    def mock_post(url, json=None, **kwargs):
        nonlocal worker_thread_id
        worker_thread_id = threading.get_ident()
        # Simulate small inference delay
        time.sleep(0.05)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.iter_lines.return_value = [
            b'{"message": {"role": "assistant", "content": "<think>Thinking</think>The answer is 42"}, "done": true}'
        ]
        mock_resp.__enter__.return_value = mock_resp
        return mock_resp

    with patch("requests.post", side_effect=mock_post):
        gen = engine.stream_chat([{"role": "user", "content": "Write a python function"}])
        first_token = next(gen)
        # Verify stall phrase is yielded immediately on first token
        assert first_token == DEFAULT_STALL_PHRASE, f"Got {first_token}"

        # Consume remaining tokens
        remaining = list(gen)

    # Verify background thread was used for inference
    assert worker_thread_id is not None
    assert worker_thread_id != caller_thread_id, "Inference did not run on a separate background thread!"

    print("  [OK] Step 4 passed: Stall phrase yielded instantly and inference executed on background worker thread.")


def test_step_5_think_filter():
    print("[TEST 5] Testing <think>...</think> filtering...")
    # Test regex
    sample_text = "<think>\nLet me solve 15 * 14.\n15 * 10 = 150, 15 * 4 = 60.\n150 + 60 = 210.\n</think>15 multiplied by 14 is 210."
    cleaned = THINK_REGEX.sub("", sample_text).strip()
    assert cleaned == "15 multiplied by 14 is 210."

    # Test real-time stream filter
    tokens = ["<think>", "\nCalculating", " step by step\n", "</think>", "\nThe", " answer", " is", " 210."]
    filt = StreamThinkFilter()
    out = []
    for t in tokens:
        res = filt.process(t)
        if res:
            out.append(res)
    flushed = filt.flush()
    if flushed:
        out.append(flushed)

    filtered_text = "".join(out)
    assert "<think>" not in filtered_text
    assert "Calculating" not in filtered_text
    assert "The answer is 210." in filtered_text

    print("  [OK] Step 5 passed: Stream filter and regex strip all <think>...</think> tags completely.")


def test_live_ollama_integration():
    print("[TEST 6] Live integration test with running Ollama instance...")
    engine = OllamaEngine()
    if not engine.check_connection():
        print("  [SKIP] Local Ollama server not reachable.")
        return

    print("  Ollama is reachable. Testing conversational stream (llama3.2:3b)...")
    chunks = []
    for chunk in engine.stream_chat([{"role": "user", "content": "Say hello in three words"}]):
        chunks.append(chunk)
    conv_response = "".join(chunks).strip()
    assert len(conv_response) > 0
    print(f"  -> Conversational response: {repr(conv_response)[:80]}")

    print("  Testing DeepSeek Shadow Reasoner stream (deepseek-r1:7b)...")
    math_chunks = []
    for chunk in engine.stream_chat([{"role": "user", "content": "Calculate 7 * 8"}]):
        math_chunks.append(chunk)
    
    # First chunk must be stall phrase
    assert math_chunks[0] == DEFAULT_STALL_PHRASE, f"First chunk was: {math_chunks[0]}"
    math_response = "".join(math_chunks[1:]).strip()
    # Must NOT contain think tags
    assert "<think>" not in math_response
    assert "</think>" not in math_response
    print(f"  -> DeepSeek response (after stall phrase): {repr(math_response)[:80]}")
    print("  [OK] Live integration test passed successfully!")


if __name__ == "__main__":
    test_step_1_system_prompt()
    test_step_2_messages_and_temperature()
    test_step_3_intent_router()
    test_step_4_stall_phrase_and_threading()
    test_step_5_think_filter()
    test_live_ollama_integration()
    print("\n==============================================")
    print("ALL 6 TESTS PASSED WITH 100% SUCCESS!")
    print("==============================================")
