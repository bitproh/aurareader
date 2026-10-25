"""
tts_engine.py - High-fidelity neural TTS engine with Kokoro-ONNX and Edge-TTS.
Features gapless audio streaming, silence trimming, lookahead prebuffering, and customizable sentence pause.
"""

import os
import sys
import time
import io
import asyncio
import threading
from typing import List, Dict, Optional, Callable, Tuple
import numpy as np
import sounddevice as sd
import soundfile as sf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "kokoro-v1.0.onnx")
VOICES_PATH = os.path.join(MODELS_DIR, "voices-v1.0.bin")

# Friendly labels for top Kokoro voices (Offline)
KOKORO_VOICES = {
    "af_sarah": "Sarah (American Female - Natural & Calm)",
    "af_bella": "Bella (American Female - Expressive)",
    "af_heart": "Heart (American Female - Warm Audiobook)",
    "af_nicole": "Nicole (American Female - Soft & Clear)",
    "af_sky": "Sky (American Female - Upbeat)",
    "am_adam": "Adam (American Male - Deep & Grounded)",
    "am_michael": "Michael (American Male - Narrator)",
    "am_eric": "Eric (American Male - Friendly)",
    "am_fenrir": "Fenrir (American Male - Cinematic)",
    "bf_emma": "Emma (British Female - Storyteller)",
    "bf_isabella": "Isabella (British Female - Elegant)",
    "bm_george": "George (British Male - Classic BBC)",
    "bm_lewis": "Lewis (British Male - Casual)",
}

# Friendly labels for Edge-TTS voices (Online Studio Neural)
EDGE_VOICES = {
    "en-US-ChristopherNeural": "Christopher (US Male - Warm & Natural)",
    "en-US-JennyNeural": "Jenny (US Female - Clear & Conversational)",
    "en-US-GuyNeural": "Guy (US Male - Classic Narrator)",
    "en-US-AriaNeural": "Aria (US Female - Expressive)",
    "en-US-EricNeural": "Eric (US Male - Casual)",
    "en-GB-SoniaNeural": "Sonia (British Female - Natural Storyteller)",
    "en-GB-RyanNeural": "Ryan (British Male - Smooth & Refined)",
}


def trim_silence(audio: np.ndarray, sr: int = 24000, threshold: float = 0.015, pad_end_ms: int = 60) -> np.ndarray:
    """
    Trims artificial silence at start and end of synthesized audio,
    adding only a short natural pause (pad_end_ms) with anti-click micro-fade.
    """
    if audio is None or len(audio) == 0:
        return audio
    mag = np.abs(audio)
    nonzero = np.where(mag > threshold)[0]
    if len(nonzero) == 0:
        return audio

    # Keep 15ms at start
    start = max(0, nonzero[0] - int(sr * 0.015))
    # Keep only pad_end_ms at end
    end = min(len(audio), nonzero[-1] + int(sr * (pad_end_ms / 1000.0)))
    trimmed = np.copy(audio[start:end])

    # 4ms fade out at end to avoid any pop/click
    fade_len = min(int(sr * 0.004), len(trimmed))
    if fade_len > 0:
        fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        trimmed[-fade_len:] *= fade

    return trimmed


class TTSEngine:
    def __init__(self):
        self.kokoro = None
        self.is_kokoro_loaded = False
        self._kokoro_lock = threading.Lock()
        self.engine_type = "kokoro"  # 'kokoro' or 'edge'
        self.sentence_gap_sec = 0.08  # Default tight natural pause

        # Optimize ONNX Runtime session for Kokoro
        self._setup_optimized_kokoro()

    def _setup_optimized_kokoro(self):
        if not (os.path.exists(MODEL_PATH) and os.path.exists(VOICES_PATH)):
            return

        with self._kokoro_lock:
            try:
                import onnxruntime as rt
                import kokoro_onnx

                # Configure multi-threaded ONNX execution
                opts = rt.SessionOptions()
                cpu_threads = max(2, min(6, os.cpu_count() or 4))
                opts.intra_op_num_threads = cpu_threads
                opts.inter_op_num_threads = 2
                opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
                opts.execution_mode = rt.ExecutionMode.ORT_SEQUENTIAL

                # Hook custom session creation
                def custom_create_session(model_path):
                    return rt.InferenceSession(model_path, sess_options=opts, providers=['CPUExecutionProvider'])

                kokoro_onnx.session.create_session = custom_create_session

                self.kokoro = kokoro_onnx.Kokoro(MODEL_PATH, VOICES_PATH)
                self.is_kokoro_loaded = True
            except Exception as e:
                print(f"Error loading optimized Kokoro: {e}")
                self.kokoro = None
                self.is_kokoro_loaded = False

    def get_available_voices(self, engine: Optional[str] = None) -> List[Tuple[str, str]]:
        eng = engine or self.engine_type
        if eng == "edge":
            return list(EDGE_VOICES.items())
        else:
            return list(KOKORO_VOICES.items())

    def synthesize(self, text: str, voice: str, speed: float = 1.0, engine: Optional[str] = None) -> Tuple[Optional[np.ndarray], int]:
        eng = engine or self.engine_type
        if eng == "edge":
            return self._synthesize_edge(text, voice, speed)
        else:
            return self._synthesize_kokoro(text, voice, speed)

    def _synthesize_kokoro(self, text: str, voice: str, speed: float) -> Tuple[Optional[np.ndarray], int]:
        if not self.is_kokoro_loaded:
            self._setup_optimized_kokoro()
            if not self.is_kokoro_loaded:
                # Fallback to Edge-TTS if kokoro unavailable
                return self._synthesize_edge(text, "en-US-ChristopherNeural", speed)

        try:
            lang = "en-gb" if voice.startswith("b") else "en-us"
            # sentence_pause controls internal Kokoro gap
            gap = max(0.02, min(0.3, self.sentence_gap_sec))
            audio, sr = self.kokoro.create(
                text=text,
                voice=voice,
                speed=float(speed),
                lang=lang,
                sentence_pause=gap,
                clause_pause=0.08
            )
            # Post-trim trailing silence with exact target padding
            pad_ms = int(self.sentence_gap_sec * 1000)
            trimmed = trim_silence(audio, sr, pad_end_ms=pad_ms)
            return trimmed, sr
        except Exception as e:
            print(f"Kokoro synthesis error: {e}")
            return None, 24000

    def _synthesize_edge(self, text: str, voice: str, speed: float) -> Tuple[Optional[np.ndarray], int]:
        try:
            import edge_tts

            # Speed string e.g. "+20%" or "-10%"
            speed_pct = int(round((speed - 1.0) * 100))
            speed_str = f"{speed_pct:+d}%"

            async def _run():
                communicate = edge_tts.Communicate(text, voice, rate=speed_str)
                data = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        data += chunk["data"]
                return data

            raw_bytes = asyncio.run(_run())
            if not raw_bytes:
                return None, 24000

            audio, sr = sf.read(io.BytesIO(raw_bytes), dtype="float32")
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)

            # Trim silence to exact target gap
            pad_ms = int(self.sentence_gap_sec * 1000)
            trimmed = trim_silence(audio, sr, pad_end_ms=pad_ms)
            return trimmed, sr
        except Exception as e:
            print(f"Edge-TTS synthesis error: {e}")
            return None, 24000


class AudioPlayer:
    """
    High-performance audio player with:
    - Pre-buffering pipeline that stays ahead of playback
    - Gapless continuous audio streaming via sounddevice.OutputStream
    - Real-time sentence callbacks for visual sync
    """

    def __init__(self, tts_engine: TTSEngine):
        self.engine = tts_engine
        self.sentences: List[str] = []
        self.current_index: int = 0
        self.is_playing: bool = False
        self.is_paused: bool = False
        self.stop_requested: bool = False

        self.voice: str = "af_sarah"
        self.speed: float = 1.0
        self.engine_type: str = "kokoro"

        # Callbacks
        self.on_sentence_started: Optional[Callable[[int, str], None]] = None
        self.on_state_changed: Optional[Callable[[bool, bool], None]] = None
        self.on_finished: Optional[Callable[[], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None
        self.on_buffering_status: Optional[Callable[[str], None]] = None

        # Prebuffered cache: {sentence_idx: (audio_array, sample_rate)}
        self._cache: Dict[int, Tuple[np.ndarray, int]] = {}
        self._cache_lock = threading.Lock()

        # Threads & synchronization
        self._player_thread: Optional[threading.Thread] = None
        self._prebuffer_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._pause_cond = threading.Condition(self._lock)
        self._output_stream: Optional[sd.OutputStream] = None

    def set_content(self, sentences: List[str], start_index: int = 0):
        self.stop()
        with self._lock:
            self.sentences = sentences
            self.current_index = max(0, min(start_index, len(sentences) - 1)) if sentences else 0
            with self._cache_lock:
                self._cache.clear()
        self._start_prebuffering()

    def set_engine(self, engine_type: str):
        if self.engine_type != engine_type:
            self.engine_type = engine_type
            self.engine.engine_type = engine_type
            with self._cache_lock:
                self._cache.clear()
            self._start_prebuffering()

    def set_voice(self, voice: str):
        if self.voice != voice:
            self.voice = voice
            with self._cache_lock:
                self._cache.clear()
            self._start_prebuffering()

    def set_speed(self, speed: float):
        if abs(self.speed - speed) > 0.05:
            self.speed = speed
            with self._cache_lock:
                self._cache.clear()
            self._start_prebuffering()

    def set_sentence_gap(self, gap_sec: float):
        self.engine.sentence_gap_sec = gap_sec
        # Clear cache so new gap applies
        with self._cache_lock:
            self._cache.clear()
        self._start_prebuffering()

    def _start_prebuffering(self):
        """Starts or wakes up the background lookahead synthesizer thread."""
        if not self.sentences:
            return
        if self._prebuffer_thread is None or not self._prebuffer_thread.is_alive():
            self._prebuffer_thread = threading.Thread(target=self._run_prebuffer_loop, daemon=True)
            self._prebuffer_thread.start()

    def _run_prebuffer_loop(self):
        """Proactively synthesizes 3-4 sentences ahead of current index."""
        while not self.stop_requested:
            if not self.sentences:
                time.sleep(0.3)
                continue

            with self._lock:
                target_idx = self.current_index

            # Synthesize window: [target_idx, target_idx + 3]
            needed_idx = None
            for offset in range(4):
                idx = target_idx + offset
                if idx < len(self.sentences):
                    with self._cache_lock:
                        if idx not in self._cache:
                            needed_idx = idx
                            break

            if needed_idx is not None and not self.stop_requested:
                text = self.sentences[needed_idx]
                audio, sr = self.engine.synthesize(text, self.voice, self.speed, self.engine_type)
                if audio is not None:
                    with self._cache_lock:
                        self._cache[needed_idx] = (audio, sr)
                        # Keep max 8 items in cache
                        keys_to_remove = [k for k in self._cache if k < target_idx - 1 or k > target_idx + 6]
                        for k in keys_to_remove:
                            del self._cache[k]
                time.sleep(0.01)
            else:
                time.sleep(0.2)

    def play(self, from_index: Optional[int] = None):
        if from_index is not None:
            self.current_index = max(0, min(from_index, len(self.sentences) - 1)) if self.sentences else 0

        if self.is_paused:
            with self._lock:
                self.is_paused = False
                self._pause_cond.notify_all()
            if self.on_state_changed:
                self.on_state_changed(True, False)
            return

        if self.is_playing:
            return

        if not self.sentences or self.current_index >= len(self.sentences):
            return

        self.stop_requested = False
        self.is_playing = True
        self.is_paused = False

        if self.on_state_changed:
            self.on_state_changed(True, False)

        self._start_prebuffering()

        self._player_thread = threading.Thread(target=self._run_gapless_playback_loop, daemon=True)
        self._player_thread.start()

    def pause(self):
        if self.is_playing and not self.is_paused:
            with self._lock:
                self.is_paused = True
            if self._output_stream:
                try:
                    self._output_stream.stop()
                except Exception:
                    pass
            if self.on_state_changed:
                self.on_state_changed(True, True)

    def resume(self):
        if self.is_playing and self.is_paused:
            if self._output_stream:
                try:
                    self._output_stream.start()
                except Exception:
                    pass
            with self._lock:
                self.is_paused = False
                self._pause_cond.notify_all()
            if self.on_state_changed:
                self.on_state_changed(True, False)

    def stop(self):
        self.stop_requested = True
        with self._lock:
            self.is_paused = False
            self._pause_cond.notify_all()
        if self._output_stream:
            try:
                self._output_stream.abort()
                self._output_stream.close()
            except Exception:
                pass
            self._output_stream = None

        if self._player_thread and self._player_thread.is_alive():
            self._player_thread.join(timeout=1.0)

        self.is_playing = False
        self.is_paused = False
        if self.on_state_changed:
            self.on_state_changed(False, False)

    def skip_to(self, sentence_index: int):
        was_playing = self.is_playing and not self.is_paused
        self.stop()
        self.current_index = max(0, min(sentence_index, len(self.sentences) - 1)) if self.sentences else 0
        if was_playing:
            self.play()

    def _run_gapless_playback_loop(self):
        """
        Gapless continuous audio streaming loop.
        Maintains single open OutputStream and streams audio chunks seamlessly.
        """
        current_sample_rate = 24000
        try:
            # Initialize open audio stream
            self._output_stream = sd.OutputStream(
                samplerate=current_sample_rate,
                channels=1,
                dtype="float32",
                blocksize=1024
            )
            self._output_stream.start()

            while self.current_index < len(self.sentences) and not self.stop_requested:
                # Handle pause
                with self._lock:
                    while self.is_paused and not self.stop_requested:
                        self._pause_cond.wait()

                if self.stop_requested:
                    break

                idx = self.current_index
                text = self.sentences[idx]

                # Check if audio is already buffered
                audio_data = None
                sr = current_sample_rate
                with self._cache_lock:
                    if idx in self._cache:
                        audio_data, sr = self._cache[idx]

                # If not buffered, synthesize immediately
                if audio_data is None:
                    if self.on_buffering_status:
                        self.on_buffering_status(f"Synthesizing sentence {idx + 1}...")
                    audio_data, sr = self.engine.synthesize(text, self.voice, self.speed, self.engine_type)
                    if audio_data is not None:
                        with self._cache_lock:
                            self._cache[idx] = (audio_data, sr)

                if self.stop_requested:
                    break

                # Restart stream if sample rate changed (e.g. between engines)
                if sr != current_sample_rate:
                    current_sample_rate = sr
                    self._output_stream.stop()
                    self._output_stream.close()
                    self._output_stream = sd.OutputStream(
                        samplerate=current_sample_rate,
                        channels=1,
                        dtype="float32",
                        blocksize=1024
                    )
                    self._output_stream.start()

                # Notify UI of active sentence starting
                if self.on_sentence_started:
                    self.on_sentence_started(idx, text)

                # Stream audio directly into open hardware stream
                if audio_data is not None and len(audio_data) > 0:
                    chunk_size = 2048
                    pos = 0
                    total_samples = len(audio_data)

                    while pos < total_samples and not self.stop_requested:
                        # Check pause
                        with self._lock:
                            while self.is_paused and not self.stop_requested:
                                self._pause_cond.wait()

                        if self.stop_requested:
                            break

                        chunk = audio_data[pos : pos + chunk_size]
                        if len(chunk) < chunk_size:
                            # Pad last small chunk
                            padded = np.zeros(chunk_size, dtype=np.float32)
                            padded[:len(chunk)] = chunk
                            self._output_stream.write(padded)
                        else:
                            self._output_stream.write(chunk)

                        pos += chunk_size

                if self.stop_requested:
                    break

                # Advance to next sentence
                self.current_index += 1

            if not self.stop_requested and self.current_index >= len(self.sentences):
                if self.on_finished:
                    self.on_finished()

        except Exception as e:
            print(f"Playback exception: {e}")
            if self.on_error:
                self.on_error(str(e))
        finally:
            if self._output_stream:
                try:
                    self._output_stream.stop()
                    self._output_stream.close()
                except Exception:
                    pass
                self._output_stream = None
            self.is_playing = False
            self.is_paused = False
            if self.on_state_changed:
                self.on_state_changed(False, False)
