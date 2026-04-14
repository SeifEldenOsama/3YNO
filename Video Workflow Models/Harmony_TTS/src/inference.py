from __future__ import annotations
import os
import random
import wave
from pathlib import Path


GEMINI_VOICES = {
    "Puck":      "male",
    "Charon":    "male",
    "Orus":      "male",
    "Achird":    "male",
    "Enceladus": "male",
    "Zephyr":    "female",
    "Leda":      "female",
    "Kore":      "female",
    "Aoede":     "female",
    "Gacrux":    "female",
    "Sulafat":   "female",
}


def load_api_keys() -> list[str]:
    """Load all Gemini API keys from GEMINI_API_KEYS env var (comma-separated)."""
    raw = os.environ.get("GEMINI_API_KEYS", "") or os.environ.get("GEMINI_API_KEY", "")
    keys = [k.strip() for k in raw.split(",") if k.strip()]
    if not keys:
        raise RuntimeError("No Gemini API keys found. Set GEMINI_API_KEYS in .env")
    return keys


def extract_voice_name(description: str) -> str:
    """Extract the Gemini voice name — always the first word of the description.

    Expected format: "Aoede A warm female voice..."
    Falls back to 'Aoede' if the first word is not a valid voice name.
    """
    first_word = description.strip().split()[0] if description.strip() else ""
    if first_word in GEMINI_VOICES:
        return first_word
    print(f"WARNING: '{first_word}' is not a valid Gemini voice name. Falling back to 'Aoede'.")
    return "Aoede"


def pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1, sample_width: int = 2) -> bytes:
    """Convert raw PCM bytes to WAV format."""
    import io
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()


def _call_gemini(api_key: str, text: str, voice_name: str) -> bytes:
    """Make one Gemini TTS API call. Returns WAV bytes."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=text,
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name,
                    )
                )
            )
        ),
    )
    audio_data = response.candidates[0].content.parts[0].inline_data.data
    return pcm_to_wav(audio_data)


def generate_parallel(requests: list[dict]) -> list[bytes]:
    """
    Generate multiple TTS clips in parallel, each using a different random API key.
    requests: list of {"text": str, "description": str}
    returns: list of WAV bytes in the same order
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    keys = load_api_keys()
    if len(keys) < len(requests):
        # If fewer keys than requests, allow reuse
        selected_keys = [keys[i % len(keys)] for i in range(len(requests))]
        random.shuffle(selected_keys)
    else:
        selected_keys = random.sample(keys, len(requests))

    print(f"Generating {len(requests)} voices in parallel with {len(selected_keys)} different keys...")

    def _generate_one(idx: int, req: dict, key: str) -> tuple[int, bytes]:
        voice_name = extract_voice_name(req["description"])
        print(f"  [{idx+1}] key ...{key[-6:]} | voice: {voice_name}")
        try:
            result = _call_gemini(api_key=key, text=req["text"], voice_name=voice_name)
            print(f"  [{idx+1}] Done ✓")
            return idx, result
        except Exception as e:
            print(f"  [{idx+1}] Failed with key ...{key[-6:]}: {e}. Retrying with fallback...")
            # Fallback: try remaining keys
            remaining = [k for k in keys if k != key]
            random.shuffle(remaining)
            for fallback_key in remaining:
                try:
                    result = _call_gemini(api_key=fallback_key, text=req["text"], voice_name=voice_name)
                    print(f"  [{idx+1}] Done with fallback key ...{fallback_key[-6:]} ✓")
                    return idx, result
                except Exception as fe:
                    print(f"  [{idx+1}] Fallback key ...{fallback_key[-6:]} also failed: {fe}")
            raise RuntimeError(f"All keys failed for request {idx+1}")

    results = [None] * len(requests)
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = {
            executor.submit(_generate_one, i, req, key): i
            for i, (req, key) in enumerate(zip(requests, selected_keys))
        }
        for future in as_completed(futures):
            idx, wav_bytes = future.result()
            results[idx] = wav_bytes

    return results


def generate_with_fallback(text: str, description: str) -> bytes:
    """Try API keys randomly. On 429 quota error, wait then retry. On other errors, switch key."""
    import time
    import re

    keys = load_api_keys()
    voice_name = extract_voice_name(description)
    print(f"Using Gemini voice: {voice_name} | Available keys: {len(keys)}")

    remaining = keys.copy()
    random.shuffle(remaining)

    last_error = None
    while remaining:
        key = remaining.pop(0)
        try:
            print(f"Trying API key ...{key[-6:]}")
            result = _call_gemini(api_key=key, text=text, voice_name=voice_name)
            print(f"Success with key ...{key[-6:]}")
            return result
        except Exception as e:
            last_error = e
            error_str = str(e)

            # On 429 quota exhausted — extract retry delay and wait
            if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                retry_match = re.search(r"retry[^0-9]*([0-9]+(?:\.[0-9]+)?)\s*s", error_str, re.IGNORECASE)
                wait_secs = float(retry_match.group(1)) if retry_match else 45.0
                wait_secs = min(wait_secs + 2, 65.0)  # add 2s buffer, cap at 65s
                print(f"Quota exceeded on key ...{key[-6:]}. Waiting {wait_secs:.1f}s before retrying with another key...")
                time.sleep(wait_secs)
                # Put remaining keys back in pool and try again
                if not remaining:
                    remaining = [k for k in keys if k != key]
                    random.shuffle(remaining)
            else:
                print(f"Key ...{key[-6:]} failed: {e}. Switching to another key...")

    raise RuntimeError(f"All {len(keys)} API keys failed. Last error: {last_error}")


class GeminiTTSInference:
    def __init__(self, api_key: str | None = None):
        # api_key param kept for compatibility but keys are loaded from env
        pass

    def generate(
        self,
        text:        str,
        description: str,
        output_path: str = "output.wav",
    ) -> str:
        wav_bytes = generate_with_fallback(text=text, description=description)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(wav_bytes)
        print(f"Audio saved to: {output_path}")
        return output_path

    def generate_bytes(self, text: str, description: str) -> bytes:
        return generate_with_fallback(text=text, description=description)