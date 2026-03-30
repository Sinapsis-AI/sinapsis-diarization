# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import torch
from nemo.collections.asr.models import SortformerEncLabelModel

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_SORTFORMER_MODEL,
    SUPPORTED_SORTFORMER_MODELS,
)
from sinapsis_diarization.pipelines.base_processors import DiarizationEngine


class SortformerEngine(DiarizationEngine):
    def __init__(
        self,
        model_name: SUPPORTED_SORTFORMER_MODELS = DEFAULT_SORTFORMER_MODEL,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
    ):
        self.model_name = model_name
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size_in_secs)

    def _load_model_weights(self):
        model = SortformerEncLabelModel.from_pretrained(self.model_name, map_location=self.device).to(self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def _inference(self, audio_path: str) -> list[list[str]]:
        """Diarizes an audio as a list of speaker turns 'start end speaker_label'

        Args:
            audio_path (str): path to audio

        Returns:
            list[list[str]]: list of speaker turns
        """
        segments = self.model.diarize(
            audio=[audio_path],
            batch_size=1,
        )
        return segments

    @torch.inference_mode()
    def get_diarization_probs(self, audio_path: str) -> np.ndarray:
        """Extracts frame-level speaker probabilities

        Args:
            audio_path (str): path to audio

        Returns:
            np.ndarray: speaker probabilities by time segments
        """
        _, probs_list = self.model.diarize(audio=[audio_path], batch_size=1, include_tensor_outputs=True)
        # Squeeze to [Time, Speaker_Slots]
        return probs_list[0].cpu().numpy().squeeze()
