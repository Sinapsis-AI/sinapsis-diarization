# -*- coding: utf-8 -*-
import gc
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import nltk
import numpy as np
import torch
from sinapsis_core.data_containers._asr_base_models import (
    DiarizedTranscript,
    SpeakerTurn,
    SpeechConstantKeys,
    TranscriptSegment,
)
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_diarization.base_models.diarization_output import (
    AudioConfig,
    SpeechChunk,
)
from sinapsis_diarization.pipelines.context_manager import ManagedAudio


class BaseSpeechEngine(ABC):
    """Base for Speech pipelines"""

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size: int = -1,
    ):
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.pad_samples = int(0.4 * self.sample_rate)
        self.model: Any | None = self._load_model_weights()

        self.chunk_size = chunk_size
        self.use_vad = chunk_size > 0
        self._vad_model: Any | None = None
        self._vad_utils: Any | None = None

    @abstractmethod
    def _load_model_weights(self):
        """Each child defines HOW to load its specific model (Whisper, NeMo, etc.)."""
        return None

    def _load_vad_model(self):
        if self._vad_model is None:
            self._vad_model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad", model="silero_vad", trust_repo=True
            )
            self._vad_utils = utils
            self._vad_model.to(self.device)

    def _get_vad_chunks(self, audio: str | np.ndarray):
        """
        Runs VAD on the enhanced 16k buffer. Instead of chunking at a fixed interval,
        or instead of using a context window, we split according to sentences to avoid
        cutoffs.
        """
        if self.use_vad:
            self._load_vad_model()

        get_vad_ts = self._vad_utils[0] if self._vad_utils is not None else None
        if get_vad_ts is None:
            return []

        with ManagedAudio(audio, sample_rate=self.sample_rate, return_path=False) as audio_content:
            speech_ts = get_vad_ts(
                torch.from_numpy(audio_content).to(self.device),
                self._vad_model,
                sampling_rate=self.sample_rate,
            )

            return [
                SpeechChunk(
                    start=round(ts[SpeechConstantKeys.start] / self.sample_rate, 2),
                    end=round(ts["end"] / self.sample_rate, 2),
                    audio=audio_content[ts["start"] : ts["end"]],
                )
                for ts in speech_ts
            ]

    @classmethod
    def find_smart_split(cls, audio_data: np.ndarray, sr: int, search_start: int, search_end: int) -> int:
        """Finds the quietest 50ms window. Returns the global index.
        We would normally only fall back to this if VAD chunks are too long.

        Args:
            audio_data (np.ndarray): audio
            sr (int): resolution
            search_start (int): search start
            search_end (int): search end

        Returns:
            int: window
        """
        hop_length = int(sr * 0.05)  # 50ms resolution

        # Safety: Stay within array bounds
        search_start = max(0, search_start)
        search_end = min(len(audio_data), search_end)
        search_area = audio_data[search_start:search_end]

        if len(search_area) < hop_length:
            return search_start

        min_energy = float("inf")
        best_local_offset = 0

        # Scan the search area for the lowest RMS energy
        for i in range(0, len(search_area) - hop_length, hop_length):
            window = search_area[i : i + hop_length]
            energy = np.sqrt(np.mean(window**2))

            if energy < min_energy:
                min_energy = energy
                best_local_offset = i

        return search_start + best_local_offset

    def group_segments(self, chunks: list[SpeechChunk], config: AudioConfig = AudioConfig()) -> list[SpeechChunk]:
        """Groups and splits SpeechChunks into model-ready sizes using Pydantic models.

        Args:
            chunks (list[SpeechChunk]): list of chunks
            config (AudioConfig, optional): configuration of the audio. Defaults to AudioConfig().

        Returns:
            list[SpeechChunk]: list of chunks

        Yields:
            Iterator[list[SpeechChunk]]: list of chunks
        """
        max_samples = int(config.max_secs * config.sample_rate)
        search_samples = int(config.search_window * config.sample_rate)

        def get_safe_chunks():
            for chunk in chunks:
                s_ptr = 0  # Pointer in the current chunk.audio
                total_samples = len(chunk.audio)

                while (total_samples - s_ptr) > max_samples:
                    # 1. Define the search boundaries relative to s_ptr
                    # We want to split near s_ptr + max_samples
                    search_end = s_ptr + max_samples
                    search_start = search_end - search_samples

                    # 2. find_smart_split returns the absolute index in chunk.audio
                    split_idx = self.find_smart_split(chunk.audio, config.sample_rate, search_start, search_end)

                    # 3. Yield exactly from s_ptr to split_idx (No gaps)
                    yield SpeechChunk(
                        start=chunk.start + (s_ptr / config.sample_rate),
                        end=chunk.start + (split_idx / config.sample_rate),
                        audio=chunk.audio[s_ptr:split_idx],
                    )

                    # 4. Advance the pointer to EXACTLY where we left off
                    s_ptr = split_idx

                # 5. Yield the final "tail" of this segment
                if s_ptr < total_samples:
                    yield SpeechChunk(
                        start=chunk.start + (s_ptr / config.sample_rate),
                        end=chunk.end,
                        audio=chunk.audio[s_ptr:],
                    )

        # Accumulate segments into efficient chunks
        final_groups: list[SpeechChunk] = []
        for safe in get_safe_chunks():
            if not final_groups:
                final_groups.append(safe)
            else:
                last = final_groups[-1]
                current_duration = safe.end - last.start

                if current_duration <= config.max_secs:
                    # Merge logic: extend end time and append audio
                    last.end = safe.end
                    last.audio = np.concatenate([last.audio, safe.audio])
                else:
                    # Too big to merge, start a new group
                    final_groups.append(safe)

        return final_groups

    def prepare_audio(self, audio_array_or_path: str | np.ndarray, config: AudioConfig) -> list[SpeechChunk]:
        """Calls for division of audio in chunks if chunk size is defined, else
        it returns the audio as one chunk.

        Args:
            audio_array_or_path (str | np.ndarray): audio
            config (AudioConfig): audio configuration

        Returns:
            list[SpeechChunk]: list of audio chunks
        """
        if self.use_vad:
            segments = self._get_vad_chunks(audio_array_or_path)
            return self.group_segments(segments, config=config)

        return [
            SpeechChunk(
                start=0.0,
                end=len(audio_array_or_path) / self.sample_rate,
                audio=audio_array_or_path,
            )
        ]

    def make_sentence_tokenizer(self):
        try:
            nltk.download("punkt_tab")
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
        return nltk.data.load("tokenizers/punkt/english.pickle")

    def flush(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def slice_audio(audio: np.ndarray, start: int, end: int, sample_rate: int, pad_samples: int) -> np.ndarray:
        """Slicing from audio_arr to allow padding beyond the chunk's own audio

        Args:
            audio (np.ndarray): audio array
            start (int): start
            end (int): end
            sample_rate (int): sample rate
            pad_samples (int): Padding

        Returns:
            np.ndarray: sliced audio
        """
        s_idx = max(0, int(start * sample_rate) - pad_samples)
        e_idx = min(len(audio), int(end * sample_rate) + pad_samples)

        return audio[s_idx:e_idx]


class ASREngine(BaseSpeechEngine):
    """Base for ASR pipelines"""

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size: int = -1,
    ):
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size)
        self.sentence_tokenizer = self.make_sentence_tokenizer()

    @abstractmethod
    def _inference(self, audio: np.ndarray) -> str:
        """Each model implements its own specific call logic."""

    @abstractmethod
    def get_asr_timestamps(self, audio: np.ndarray):
        """Each model implements its own specific call logic."""

    @torch.no_grad()
    def transcribe(self, audio_path_or_array: str | np.ndarray, config: AudioConfig = AudioConfig()) -> str:
        """Transcribes each chunk of audio and joins them

        Args:
            audio_path_or_array (str | np.ndarray): path to audio or audio array
            config (AudioConfig, optional): configuration for audio. Defaults to AudioConfig().

        Returns:
            str: transcribed text
        """
        with ManagedAudio(
            audio_path_or_array,
            sample_rate=self.sample_rate,
            return_path=False,
        ) as audio_arr:
            grouped_segments = self.prepare_audio(audio_arr, config)
            if not grouped_segments:
                sinapsis_logger.warning("No segments to transcribe.")
                return ""

            transcript_parts = []

            for chunk in grouped_segments:
                sliced_audio = self.slice_audio(
                    audio=audio_arr,
                    start=int(chunk.start),
                    end=int(chunk.end),
                    sample_rate=self.sample_rate,
                    pad_samples=self.pad_samples,
                )
                text = self._inference(sliced_audio)
                if text.strip():
                    transcript_parts.append(text.strip())

        return " ".join(transcript_parts)


class DiarizationEngine(BaseSpeechEngine):
    """Base for diarization pipelines"""

    @abstractmethod
    def _inference(self, audio_path: Path | str):
        pass

    def diarize(self, audio_path_or_array, config: AudioConfig = AudioConfig()) -> list[SpeakerTurn]:
        """Diarize the audio by chunks

        Args:
            audio_path_or_array (str | np.ndarray): path to audio or audio array
            config (AudioConfig, optional): configuration for audio. Defaults to AudioConfig().

            list[SpeakerTurn]: diarized speaker
        """
        with ManagedAudio(audio_path_or_array, sample_rate=self.sample_rate, return_path=False) as audio_arr:
            grouped_segments = self.prepare_audio(audio_arr, config)
            if not grouped_segments:
                sinapsis_logger.warning("No segments to transcribe.")
                return []

            # Sortformer only accepts paths
            speaker_chunks = []
            for chunk in grouped_segments:
                # Slicing from audio_arr to allow padding beyond the chunk's own audio
                sliced_audio = self.slice_audio(
                    audio=audio_arr,
                    start=int(chunk.start),
                    end=int(chunk.end),
                    sample_rate=self.sample_rate,
                    pad_samples=self.pad_samples,
                )
                with ManagedAudio(
                    sliced_audio,
                    sample_rate=self.sample_rate,
                    return_path=True,
                ) as audio_path:
                    segment_speaker_chunks = self._inference(audio_path)
                    speaker_chunks.extend(segment_speaker_chunks)

        # Squeeze to [Time, Speaker_Slots]
        speakers_order = get_interleaved_turns(speaker_chunks)

        return speakers_order

    @abstractmethod
    def get_diarization_probs(self, audio_path_or_array) -> np.ndarray:
        pass


def get_interleaved_turns(raw_output: list[list[str]], gap_threshold=0.5) -> list[SpeakerTurn]:
    """Transform speaker segments ["<start> <end> <speaker_tag>"] into speaker turns

    Args:
        raw_output (list[list[str]]): list of speaker segments
        gap_threshold (float, optional): _description_. Defaults to 0.5.

    Returns:
        list[SpeakerTurn]: list of turns by speaker
    """
    # 1. Parse the strings into objects
    segments = []
    for entry in raw_output[0]:
        start, end, speaker = entry.split()
        segments.append(SpeakerTurn(speaker=speaker, start_time=float(start), end_time=float(end)))
    # 2. Sort by start time (crucial for interleaving)
    segments.sort(key=lambda x: x.start_time)

    if not segments:
        return []

    # 3. Merge consecutive turns by the same speaker
    merged_turns = []
    current_turn = segments[0]

    for next_seg in segments[1:]:
        # If same speaker AND the gap is small, extend the current turn
        if next_seg.speaker == current_turn.speaker and next_seg.start_time <= current_turn.end_time + gap_threshold:
            current_turn.end_time = max(current_turn.end_time, next_seg.end_time)
        else:
            # Different speaker or big gap: finalize current turn and start new one
            merged_turns.append(current_turn)
            current_turn = next_seg

    merged_turns.append(current_turn)  # Add the last one
    return merged_turns


class BaseASRDiarizationPipeline(BaseSpeechEngine):
    """Base for joined ASR and diarization pipelines"""

    def __init__(
        self,
        asr_pipeline: str,
        diarization_pipeline: str,
        device: str,
        sample_rate: int,
        chunk_size_in_secs: int,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.pad_samples = int(0.4 * self.sample_rate)
        self.chunk_size_in_secs = chunk_size_in_secs
        self.asr_pipeline_name = asr_pipeline
        self.diarization_pipeline_name = diarization_pipeline
        self._load_model_weights()
        self.sent_detector = self.make_sentence_tokenizer()
        self.use_vad = self.chunk_size_in_secs > 0
        self._vad_model: Any | None = None
        self._vad_utils: Any | None = None

    @abstractmethod
    def init_asr_pipeline(self, asr_pipeline: str) -> ASREngine:
        pass

    @abstractmethod
    def init_diarization_pipeline(self, asr_pipeline: str) -> DiarizationEngine:
        pass

    def _load_model_weights(self):
        """Initializes the ASR and diarization pipelines"""
        self.asr_pipeline = self.init_asr_pipeline(self.asr_pipeline_name)
        self.diarization_pipeline = self.init_diarization_pipeline(self.diarization_pipeline_name)

    def flush_all(self):
        self.asr_pipeline.flush()
        self.diarization_pipeline.flush()

    def transcribed_diarization(
        self, audio_path_or_array: str | np.ndarray, config: AudioConfig = AudioConfig()
    ) -> DiarizedTranscript:
        """Transcribes, diarizes and aligns the results.

        Args:
            audio_path_or_array (str | np.ndarray): path to audio or audio array
            config (AudioConfig): configuration for audio. Defaults to AudioConfig()

        Returns:
            DiarizedTranscript: Base model with the transcript by speaker turn
        """
        word_timestamps = []
        speakers_probs = []
        with ManagedAudio(audio_path_or_array, sample_rate=self.sample_rate, return_path=False) as audio_arr:
            grouped_segments = self.prepare_audio(audio_arr, config)
            if not grouped_segments:
                sinapsis_logger.warning("No segments to transcribe.")
                return ""

            for chunk in grouped_segments:
                # Slicing from audio_arr to allow padding beyond the chunk's own audio
                sliced_audio = self.slice_audio(
                    audio=audio_arr,
                    start=int(chunk.start),
                    end=int(chunk.end),
                    sample_rate=self.sample_rate,
                    pad_samples=self.pad_samples,
                )
                word_timestamps.extend(self.asr_pipeline.get_asr_timestamps(sliced_audio))

                with ManagedAudio(
                    sliced_audio,
                    sample_rate=self.sample_rate,
                    return_path=True,
                ) as audio_path:
                    segment_speaker_probs = self.diarization_pipeline.get_diarization_probs(audio_path)
                    speakers_probs.extend(segment_speaker_probs)

        # 2. Tokenize into sentences
        sentences = self._tokenize_sentences(word_timestamps)

        # 3. Align and return
        return self._align_sentences_to_speakers(sentences, word_timestamps, np.asarray(speakers_probs))

    def _tokenize_sentences(self, word_timestamps: list[dict]) -> list[str]:
        """Joins words and splits them into logical sentences.

        Args:
            word_timestamps (list[dict]): timestamps by word

        Returns:
            list[str]: list of sentences
        """
        full_text = " ".join(w["word"] for w in word_timestamps)
        return self.sent_detector.tokenize(full_text)

    def _align_sentences_to_speakers(
        self,
        sentences: list[str],
        word_timestamps: list[dict],
        speaker_probs: np.ndarray,
    ) -> DiarizedTranscript:
        """Maps sentence boundaries to the most probable speaker in that timeframe.

        Args:
            sentences (list[str]): list of sentences
            word_timestamps (list[dict]): timestamps by word
            speaker_probs (np.ndarray): probabilities of speaker in time windows

        Returns:
            DiarizedTranscript: Transcript with speaker labels
        """

        final_script = DiarizedTranscript(segments=[])
        word_idx = 0

        for sent in sentences:
            sent_words = sent.split()
            if word_idx >= len(word_timestamps):
                break

            # Time boundaries
            start_time = word_timestamps[word_idx]["start"]
            last_idx = min(word_idx + len(sent_words) - 1, len(word_timestamps) - 1)
            end_time = word_timestamps[last_idx]["end"]

            # Identify dominant speaker
            speaker_label = self._get_dominant_speaker(start_time, end_time, speaker_probs)
            final_script.segments.append(
                TranscriptSegment(
                    speaker=speaker_label,
                    start_time=start_time,
                    end_time=end_time,
                    text=sent.strip(),
                )
            )

            word_idx += len(sent_words)

        return final_script

    def _get_dominant_speaker(self, start: float, end: float, probs: np.ndarray) -> str:
        """Calculates the speaker with the highest probability sum in a time window.

        Args:
            start (float): start of segment
            end (float): end of segment
            probs (np.ndarray): probabilities by speaker

        Returns:
            str: label of dominant speaker
        """
        # Sortformer uses 80ms frames
        s_frame = int(start / 0.08)
        e_frame = int(end / 0.08)

        segment = probs[s_frame : max(e_frame, s_frame + 1)]
        winner = np.argmax(np.sum(segment, axis=0)) if segment.size > 0 else 0

        return f"SPEAKER {winner}"


class BaseEmotionEngine(ABC):
    """Base for emotion recognition pipelines"""

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
    ):
        self.device = device
        self.sample_rate = sample_rate
        self.model: Any | None = self._load_model_weights()

    @abstractmethod
    def _load_model_weights(self):
        """Each child defines HOW to load its specific model (Whisper, NeMo, etc.)."""
        return None
