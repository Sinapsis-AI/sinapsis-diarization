# -*- coding: utf-8 -*-

import os

from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_diarization.base_models.diarization_output import AudioConfig
from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_PARAKEET_MODEL,
    DEFAULT_PYANNOTE_MODEL,
    DEFAULT_SORTFORMER_MODEL,
    DEFAULT_WHISPERX_MODEL,
)
from sinapsis_diarization.pipelines.asr_diarization import ASRDiarizationPipeline
from sinapsis_diarization.pipelines.whisperx_pipeline import WhisperxASRDiarizationPipeline

NO_MODEL = "NO_MODEL_NAME"


def run_asr_diarization(
    audio_path: str,
    asr_model: str,
    diarization_model: str,
    chunk_size: int = -1,
    asr_model_name: str = NO_MODEL,
    diarization_model_name: str = NO_MODEL,
    device: str = "cuda",
    sample_rate: int = 16000,
    output_dir: str = "results",
    num_speakers: int = 2,
):
    model: ASRDiarizationPipeline
    if asr_model == "parakeet":
        asr_model_name_input = asr_model_name if asr_model_name != NO_MODEL else DEFAULT_PARAKEET_MODEL
    else:
        raise NotImplementedError(f"{asr_model} is not an implemented model")
    if diarization_model == "pyannote":
        diarization_model_name_input = (
            diarization_model_name if diarization_model_name != NO_MODEL else DEFAULT_PYANNOTE_MODEL
        )
        model = ASRDiarizationPipeline(
            asr_pipeline=asr_model,
            diarization_pipeline=diarization_model,
            asr_model_name=asr_model_name_input,
            diarization_model_name=diarization_model_name_input,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size,
            num_speakers=num_speakers,
        )
    elif diarization_model == "sortformer":
        diarization_model_name_input = (
            diarization_model_name if diarization_model_name != NO_MODEL else DEFAULT_SORTFORMER_MODEL
        )
        model = ASRDiarizationPipeline(
            asr_pipeline=asr_model,
            diarization_pipeline=diarization_model,
            asr_model_name=asr_model_name_input,
            diarization_model_name=diarization_model_name_input,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size,
            num_speakers=num_speakers,
        )
    else:
        raise NotImplementedError(f"{diarization_model} is not an implemented model")
    audio_config = AudioConfig(sample_rate)
    transcription = model.transcribed_diarization(audio_path, audio_config)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"transcribed_diarization_{asr_model}_{diarization_model}.txt")
    sinapsis_logger.info(f"Results saved at {output_file}")
    with open(output_file, "w") as output_obj:
        for turn in transcription.segments:
            output_obj.write(f"{turn.speaker}: {turn.text}\n")


def run_whisperx_asr_diarization(
    audio_path: str,
    chunk_size: int = -1,
    model_name: str = NO_MODEL,
    device: str = "cuda",
    sample_rate: int = 16000,
    output_dir: str = "results",
    min_speakers: int = 2,
    max_speakers: int = 2,
):
    model: WhisperxASRDiarizationPipeline = WhisperxASRDiarizationPipeline(
        asr_model_name=model_name if model_name != NO_MODEL else DEFAULT_WHISPERX_MODEL,
        device=device,
        sample_rate=sample_rate,
        chunk_size_in_secs=chunk_size,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    )

    audio_config = AudioConfig(sample_rate)
    transcription = model.transcribed_diarization(audio_path, audio_config)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, "transcribed_diarization_whisperx.txt")
    sinapsis_logger.info(f"Results saved at {output_file}")
    with open(output_file, "w") as output_obj:
        for turn in transcription.segments:
            output_obj.write(f"{turn.speaker}: {turn.text}\n")
