# -*- coding: utf-8 -*-

import importlib
import os

from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_diarization.base_models.diarization_output import AudioConfig
from sinapsis_diarization.base_models.supported_models import DEFAULT_CANARY_MODEL, DEFAULT_PARAKEET_MODEL
from sinapsis_diarization.pipelines.base_processors import ASREngine

NO_MODEL = "NO_MODEL_NAME"


def run_asr(
    audio_path: str,
    model: str,
    chunk_size: int = -1,
    model_name: str = NO_MODEL,
    device: str = "cuda",
    sample_rate: int = 16000,
    output_dir: str = "results",
):
    asr_model: ASREngine
    if model == "parakeet":
        module = importlib.import_module("sinapsis_diarization.pipelines.parakeet_asr")
        ParakeetASR = getattr(module, "ParakeetASR")
        asr_model = ParakeetASR(
            model_name=model_name if model_name != NO_MODEL else DEFAULT_PARAKEET_MODEL,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size,
        )
    elif model == "canary":
        module = importlib.import_module("sinapsis_diarization.pipelines.canary_asr")
        CanaryTranscriber = getattr(module, "CanaryTranscriber")
        asr_model = CanaryTranscriber(
            model_name=model_name if model_name != NO_MODEL else DEFAULT_CANARY_MODEL,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size,
        )
    else:
        raise NotImplementedError(f"{model} is not implemented")

    audio_config = AudioConfig(sample_rate)
    transcription = asr_model.transcribe(audio_path, audio_config)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"transcription_{model}.txt")
    sinapsis_logger.info(f"Results saved at {output_file}")
    with open(output_file, "w") as output_obj:
        output_obj.write(transcription)
