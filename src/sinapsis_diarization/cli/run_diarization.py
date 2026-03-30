# -*- coding: utf-8 -*-

import importlib
import os

from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_diarization.base_models.diarization_output import AudioConfig
from sinapsis_diarization.base_models.supported_models import DEFAULT_PYANNOTE_MODEL, DEFAULT_SORTFORMER_MODEL
from sinapsis_diarization.pipelines.base_processors import DiarizationEngine

NO_MODEL = "NO_MODEL_NAME"


def run_diarization(
    audio_path: str,
    model: str,
    chunk_size: int = -1,
    model_name: str = NO_MODEL,
    device: str = "cuda",
    sample_rate: int = 16000,
    output_dir: str = "results",
):
    diarization_model: DiarizationEngine
    if model == "sortformer":
        module = importlib.import_module("sinapsis_diarization.pipelines.sortformer_diarizer")
        SortformerEngine = getattr(module, "SortformerEngine")
        diarization_model = SortformerEngine(
            model_name=model_name if model_name != NO_MODEL else DEFAULT_SORTFORMER_MODEL,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size,
        )
    elif model == "pyannote":
        module = importlib.import_module("sinapsis_diarization.pipelines.pyannote_diarizer")
        PyannoteEngine = getattr(module, "PyannoteEngine")
        diarization_model = PyannoteEngine(
            model_name=model_name if model_name != NO_MODEL else DEFAULT_PYANNOTE_MODEL,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size,
        )
    else:
        raise NotImplementedError(f"{model} is not implemented")

    audio_config = AudioConfig(sample_rate)
    diarization = diarization_model.diarize(audio_path, audio_config)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = os.path.join(output_dir, f"diarization_{model}.txt")
    sinapsis_logger.info(f"Results saved at {output_file}")
    with open(output_file, "w") as output_obj:
        for turn in diarization:
            output_obj.write(f"{turn.speaker}: start:{turn.start_time}, end: {turn.end_time}\n")
