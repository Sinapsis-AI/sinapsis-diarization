# -*- coding: utf-8 -*-
from typing import Literal

from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_diarization.base_models.supported_models import (
    SUPPORTED_PARAKEET_MODELS,
    SUPPORTED_PYANNOTE_MODELS,
    SUPPORTED_SORTFORMER_MODELS,
)
from sinapsis_diarization.pipelines.base_processors import (
    ASREngine,
    BaseASRDiarizationPipeline,
    DiarizationEngine,
)
from sinapsis_diarization.pipelines.parakeet_asr import ParakeetASR
from sinapsis_diarization.pipelines.pyannote_diarizer import PyannoteEngine
from sinapsis_diarization.pipelines.sortformer_diarizer import SortformerEngine

SUPPORTED_ASR_ENGINES = Literal["parakeet"]
SUPPORTED_DIARIZER_ENGINES = Literal[
    "sortformer",
    "pyannote",
]


class ASRDiarizationPipeline(BaseASRDiarizationPipeline):
    """Define a pipeline for automatic speech recognition and diarization"""

    def __init__(
        self,
        asr_pipeline: SUPPORTED_ASR_ENGINES,
        diarization_pipeline: SUPPORTED_DIARIZER_ENGINES,
        asr_model_name: SUPPORTED_PARAKEET_MODELS,
        diarization_model_name: SUPPORTED_SORTFORMER_MODELS | SUPPORTED_PYANNOTE_MODELS,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
        num_speakers: int = 2,
    ):
        self.asr_model_name = asr_model_name
        self.diarization_model_name = diarization_model_name
        self.num_speakers = num_speakers
        super().__init__(
            asr_pipeline=asr_pipeline,
            diarization_pipeline=diarization_pipeline,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size_in_secs,
        )

    def init_asr_pipeline(self, asr_pipeline: str) -> ASREngine:
        """Choose the ASR pipeline to initialize

        Args:
            asr_pipeline (str): Name of the ASR pipeline

        Returns:
            ASREngine: ASR pipeline
        """
        if asr_pipeline == "parakeet":
            return ParakeetASR(
                model_name=self.asr_model_name,
                device=self.device,
                sample_rate=self.sample_rate,
                chunk_size_in_secs=self.chunk_size_in_secs,
            )
        else:
            sinapsis_logger.error(f"{asr_pipeline} is not an implemented ASR engine")

    def init_diarization_pipeline(self, diarization_pipeline: str) -> DiarizationEngine:
        """Choose the diarization pipeline to initialize

        Args:
            diarization_pipeline (str): Name of the diarization pipeline

        Returns:
            DiarizationEngine: Diarization pipeline
        """
        if diarization_pipeline == "sortformer":
            return SortformerEngine(
                model_name=self.diarization_model_name,
                device=self.device,
                sample_rate=self.sample_rate,
                chunk_size_in_secs=self.chunk_size_in_secs,
            )
        if diarization_pipeline == "pyannote":
            return PyannoteEngine(
                model_name=self.diarization_model_name,
                device=self.device,
                sample_rate=self.sample_rate,
                chunk_size_in_secs=self.chunk_size_in_secs,
                num_speakers=self.num_speakers,
            )
