# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
import torch
from nemo.collections.speechlm2.models import SALM

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_CANARY_MODEL,
    SUPPORTED_CANARY_MODELS,
)
from sinapsis_diarization.pipelines.base_processors import ASREngine


class CanaryTranscriber(ASREngine):
    """Pipeline for Canary ASR"""

    def __init__(
        self,
        model_name: SUPPORTED_CANARY_MODELS = DEFAULT_CANARY_MODEL,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
    ):
        self.model_name = model_name
        super().__init__(device=device, sample_rate=sample_rate, chunk_size=chunk_size_in_secs)

    def get_asr_timestamps(self, audio: np.ndarray):
        """Not implemented for Canary."""
        _ = audio
        return []

    @staticmethod
    def _normalize_device(device: torch.device) -> str:
        dev_str = str(device)

        if dev_str.startswith("cuda"):
            return dev_str if ":" in dev_str else "cuda:0"
        return "cpu"

    def _load_model_weights(self):
        device = self._normalize_device(self.device)
        asr_model = SALM.from_pretrained(self.model_name, map_location=device).to(device)
        asr_model.eval()
        return asr_model

    @torch.inference_mode()
    def _inference(self, audio: np.ndarray) -> str:
        """Transcribes an audio

        Args:
            audio (np.ndarray): audio array

        Returns:
            str: transcription
        """
        audio_tensor = torch.from_numpy(audio).to(self.device)
        if audio_tensor.ndim == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        audio_len = torch.tensor([audio_tensor.shape[1]], device=self.device)

        # Canary-specific prompt/locator API
        if self.model:
            prompts = [[{"role": "user", "content": f"Transcribe: {self.model.audio_locator_tag}"}]]

            output_ids = self.model.generate(
                prompts=prompts,
                audios=audio_tensor,
                audio_lens=audio_len,
                max_new_tokens=512,
                num_beams=1,
                bos_token_id=151643,
            )
            return self.model.tokenizer.ids_to_text(output_ids[0].cpu())
        return ""
