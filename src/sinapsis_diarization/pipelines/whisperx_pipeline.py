# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import pandas as pd
import torch
from sinapsis_core.data_containers._asr_base_models import (
    DiarizedTranscript,
    SpeechConstantKeys,
    TranscriptSegment,
)
from sinapsis_core.utils.logging_utils import sinapsis_logger
from whisperx.alignment import align, load_align_model
from whisperx.asr import load_model
from whisperx.audio import load_audio
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_WHISPERX_MODEL,
    SUPPORTED_WHISPERX_MODELS,
)
from sinapsis_diarization.helpers.env_var_keys import DiarizationEnvVars
from sinapsis_diarization.pipelines.base_processors import (
    ASREngine,
    BaseASRDiarizationPipeline,
    DiarizationEngine,
)
from sinapsis_diarization.pipelines.context_manager import ManagedAudio


class WhisperxASREngine(ASREngine):
    def __init__(
        self,
        model_name: SUPPORTED_WHISPERX_MODELS = DEFAULT_WHISPERX_MODEL,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
    ):
        self.model_name = model_name
        self.device_str = device
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size_in_secs)

    def _load_model_weights(self):
        model = load_model(
            self.model_name,
            self.device_str,
        )
        return model

    @torch.inference_mode()
    def _inference(self, audio_path: str) -> str:
        """Transcribes an audio as a single string

        Args:
            audio (np.ndarray): path to audio

        Returns:
            str: transcription
        """
        audio = load_audio(audio_path)
        if self.model:
            result = self.model.transcribe(audio, batch_size=1, language="en")
            segments_texts = [segment["text"] for segment in result["segments"]]
            return " ".join(segments_texts)
        return ""

    def get_asr_timestamps(self, audio: np.ndarray):
        """Not implemented for Whisperx"""
        _ = audio
        return

    @torch.inference_mode()
    def _inference_pipeline(self, audio_path: str) -> str:
        """Transcribes an audio as a list of segments for the full ASR Diarization pipeline

        Args:
            audio (np.ndarray): path to audio

        Returns:
            str: transcription
        """
        audio = load_audio(audio_path)
        if self.model:
            result = self.model.transcribe(audio, batch_size=1, language="en")
            return result["segments"]
        return ""


class WhisperxDiarizationEngine(DiarizationEngine):
    """Pipeline for Diarization with Whisperx"""

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
        min_speakers: int = 1,
        max_speakers: int = 2,
    ):
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size_in_secs)

    def _load_model_weights(self):
        model = DiarizationPipeline(token=DiarizationEnvVars.HF_TOKEN, device=self.device)
        return model

    @torch.inference_mode()
    def _inference(self, audio_path: str) -> list[list[str]]:
        """Diarizes an audio as a list of speaker turns 'start end speaker_label'

        Args:
            audio_path (str): path to audio

        Returns:
            list[list[str]]: list of speaker turns
        """
        audio = load_audio(audio_path)
        result: pd.DataFrame = self.model(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)  # type: ignore [misc]
        segments: list[str] = [f"{float(row.start)} {float(row.end)} {row.speaker}" for row in result.itertuples()]
        return [segments]

    @torch.inference_mode()
    def get_diarization_probs(self, audio_path: str) -> np.ndarray:
        """Not implemented for Whisperx"""
        _ = audio_path
        return

    @torch.inference_mode()
    def _inference_pipeline(self, audio_path: str) -> list[list[str]]:
        """Diarizes an audio as a DataFrame for the full ASR Diarization pipeline

        Args:
            audio_path (str): _description_

        Returns:
            list[list[str]]: _description_
        """
        audio = load_audio(audio_path)
        result: pd.DataFrame = self.model(audio, min_speakers=self.min_speakers, max_speakers=self.max_speakers)  # type: ignore [misc]
        return result


class WhisperxASRDiarizationPipeline(BaseASRDiarizationPipeline):
    """Pipeline for ASR and diarization with Whisperx"""

    def __init__(
        self,
        asr_model_name: SUPPORTED_WHISPERX_MODELS,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
        min_speakers: int = 1,
        max_speakers: int = 2,
    ):
        self.asr_model_name = asr_model_name
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        super().__init__(
            asr_pipeline="whisperx",
            diarization_pipeline="whisperx",
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size_in_secs,
        )
        # Initialize NLTK
        self.__sent_detector = self.make_sentence_tokenizer()

    def init_asr_pipeline(self, asr_pipeline: str) -> ASREngine:
        _ = asr_pipeline
        return WhisperxASREngine(
            model_name=self.asr_model_name,
            device=self.device,
            sample_rate=self.sample_rate,
            chunk_size_in_secs=self.chunk_size_in_secs,
        )

    def init_diarization_pipeline(self, diarization_pipeline: str) -> DiarizationEngine:
        _ = diarization_pipeline
        return WhisperxDiarizationEngine(
            device=self.device,
            sample_rate=self.sample_rate,
            chunk_size_in_secs=self.chunk_size_in_secs,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )

    def transcribed_diarization(self, audio_path_or_array, config):
        """Transcribes, diarizes and aligns the results.

        Args:
            audio_path_or_array (str | np.ndarray): path to audio or audio array
            config (AudioConfig): configuration for audio. Defaults to AudioConfig()

        Returns:
            DiarizedTranscript: Base model with the transcript by speaker turn
        """
        results = []
        with ManagedAudio(audio_path_or_array, sample_rate=self.sample_rate, return_path=False) as audio_arr:
            grouped_segments = self.prepare_audio(audio_arr, config)
            if not grouped_segments:
                sinapsis_logger.warning("No segments to transcribe.")
                return ""

            for chunk in grouped_segments:
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
                    audio = load_audio(audio_path)
                    raw_segments_asr = self.asr_pipeline._inference_pipeline(audio_path)
                    aligned_result = self.__align(audio, raw_segments_asr)
                    diarized_segments = self.diarization_pipeline._inference_pipeline(audio_path)
                    result = assign_word_speakers(diarized_segments, aligned_result)
                    results.extend(result["segments"])

        diarized_transcript = self.__reconstruct_sentences(results)

        self.flush_all()
        return diarized_transcript

    @torch.inference_mode()
    def __align(self, audio: np.ndarray, segments: list[dict]) -> dict:
        """Aligns transcription segments to exact word-level timestamps.

        Args:
            audio (np.ndarray): audio array
            segments (list[dict]): transcription segments

        Returns:
            dict: aligned result
        """
        model_a, metadata = load_align_model(language_code="en", device=self.device)
        result = align(
            segments,
            model_a,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

        return result

    def __reconstruct_sentences(self, segments: list[dict]) -> DiarizedTranscript:
        """Uses NLTK to group words into sentences and assigns dominant speakers.

        Args:
            segments (list[dict]): list of segments

        Returns:
            DiarizedTranscript: Transcript with speaker label
        """
        all_words = [w for seg in segments for w in seg.get("words", []) if "start" in w]
        if not all_words:
            return []

        full_text = " ".join([w["word"].strip() for w in all_words])
        sentences = self.__sent_detector.tokenize(full_text)

        word_idx = 0
        final_entries = DiarizedTranscript(segments=[])

        for sentence in sentences:
            sentence_parts = sentence.split()
            sentence_data = all_words[word_idx : word_idx + len(sentence_parts)]
            word_idx += len(sentence_parts)

            if not sentence_data:
                continue

            # Duration-based majority vote for speaker
            spk_votes: dict[str, float] = {}
            for w in sentence_data:
                spk = w.get("speaker", SpeechConstantKeys.unknown_speaker_tag)
                dur = w["end"] - w["start"]
                spk_votes[spk] = spk_votes.get(spk, 0) + dur

            dominant_spk = max(spk_votes, key=spk_votes.get) if spk_votes else None  # type: ignore [arg-type]

            final_entries.segments.append(
                TranscriptSegment(
                    speaker=dominant_spk,
                    start_time=sentence_data[0]["start"],
                    end_time=sentence_data[-1]["end"],
                    text=sentence.strip(),
                )
            )

        return self.__smooth_speakers(final_entries)

    def __smooth_speakers(self, entries: DiarizedTranscript) -> DiarizedTranscript:
        """Prevents rapid flickering between speakers for short durations.

        Args:
            entries (DiarizedTranscript): Transcript with speaker labels

        Returns:
            DiarizedTranscript: smoothed transcript
        """
        for i in range(1, len(entries.segments) - 1):
            prev_s = entries.segments[i - 1].speaker
            next_s = entries.segments[i + 1].speaker
            curr_s = entries.segments[i].speaker

            if (prev_s == next_s and curr_s != prev_s) and (
                entries.segments[i].end_time - entries.segments[i].start_time < 1.2
            ):
                entries.segments[i].speaker = prev_s
        return entries
