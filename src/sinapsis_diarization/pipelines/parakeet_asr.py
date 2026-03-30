# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_PARAKEET_MODEL,
    SUPPORTED_PARAKEET_MODELS,
)
from sinapsis_diarization.pipelines.base_processors import ASREngine


class ParakeetASR(ASREngine):
    """Pipeline for Parakeet ASR"""

    def __init__(
        self,
        model_name: SUPPORTED_PARAKEET_MODELS = DEFAULT_PARAKEET_MODEL,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
    ):
        self.model_name = model_name
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size_in_secs)

    def _load_model_weights(self):
        model = ASRModel.from_pretrained(model_name=self.model_name, map_location=self.device).to(device=self.device)
        model.eval()
        return model

    @torch.inference_mode()
    def _inference(self, audio: np.ndarray) -> str:
        """Transcribes an audio

        Args:
            audio (np.ndarray): audio array

        Returns:
            str: transcription
        """
        audio_tensor = torch.from_numpy(audio).to(self.device)
        if self.model:
            res = self.model.transcribe(audio=[audio_tensor], batch_size=1, return_hypotheses=True)
            transcription = res[0][0].text if isinstance(res[0], list) else res[0].text
            return transcription
        return ""

    def get_asr_timestamps(self, audio: np.ndarray) -> str:
        """Gets timestamps of each word from an audio

        Args:
            audio (np.ndarray): audio array

        Returns:
            str: word timestamps
        """
        audio_tensor = torch.from_numpy(audio).to(self.device)
        if self.model:
            res = self.model.transcribe(audio=[audio_tensor], batch_size=1, return_hypotheses=True, timestamps=True)
            return res[0][0].timestamp["word"] if isinstance(res[0], list) else res[0].timestamp["word"]
        return ""
