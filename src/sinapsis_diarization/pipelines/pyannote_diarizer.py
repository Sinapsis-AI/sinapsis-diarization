# -*- coding: utf-8 -*-
from typing import Any, Literal

import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_PYANNOTE_MODEL,
    SUPPORTED_PYANNOTE_MODELS,
)
from sinapsis_diarization.pipelines.base_processors import DiarizationEngine


class PyannoteEngine(DiarizationEngine):
    """
    Pipeline for Pyannote Diarization
    """

    def __init__(
        self,
        model_name: SUPPORTED_PYANNOTE_MODELS = DEFAULT_PYANNOTE_MODEL,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
        num_speakers: int = 2,
    ):
        self.model_name = model_name
        self.num_speakers = num_speakers
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size_in_secs)

    def _load_model_weights(self):
        model = Pipeline.from_pretrained(self.model_name)
        model.to(self.device)
        return model

    @torch.inference_mode()
    def _inference(self, audio_path: str) -> list[list[str]]:
        """Diarize an audio as a list of speaker turns 'start end speaker_label'

        Args:
            audio_path (str): path to audio

        Returns:
            list[list[str]]: list of speaker turns
        """
        with ProgressHook() as hook:
            if self.model:
                diarization = self.model(audio_path, hook=hook, preload=True, num_speakers=self.num_speakers)

        results = [f"{turn.start} {turn.end} {speaker}" for turn, speaker in diarization.speaker_diarization]

        return [results]

    def get_diarization_probs(self, audio_path: str) -> list:
        """Transform the diarization output of pyannote into speaker probabilities with
        0.08 second segments

        Args:
            audio_path (str): _description_

        Returns:
            list
        """
        with ProgressHook() as hook:
            if self.model:
                diarization = self.model(audio_path, hook=hook, preload=True, num_speakers=self.num_speakers)
        speaker_turns: dict[str, Any] = {}
        speaker_ids: dict[str, int] = {}
        idx = 0
        end_time = 0
        for turn, speaker in diarization.speaker_diarization:
            if speaker not in speaker_turns:
                speaker_turns[speaker] = []
                speaker_ids[speaker] = idx
                idx += 1
            speaker_turns[speaker].append([turn.start, turn.end])
            if turn.end > end_time:
                end_time = turn.end
        # ASR Diarization uses 0.08s segments
        total_probs = int(end_time / 0.08)
        # probs = np.zeros((total_probs, len(speaker_turns)))
        probs = [[0 for i in range(self.num_speakers)] for j in range(total_probs)]
        for speaker_key, turns in speaker_turns.items():
            speaker_id = speaker_ids[speaker_key]
            for turn in turns:
                segment_start_idx = int(turn[0] / 0.08)
                segment_end_idx = int(turn[1] / 0.08)
                for i in range(segment_start_idx, segment_end_idx):
                    probs[i][speaker_id] = 1
        return probs
