# -*- coding: utf-8 -*-
from sinapsis_diarization.pipelines.whisperx_pipeline import (
    WhisperxASRDiarizationPipeline,
)
from sinapsis_diarization.templates.asr_diarization_template import (
    ASRDiarizationAttributes,
    SinapsisASRDiarization,
)


class WhisperxASRDiarizationAttributes(ASRDiarizationAttributes):
    """
    diarization_model_name (str): defaults to empty string. Unused in Whisperx pipeline
    min_speakers (int): minimum speakers to predict
    max_speakers (int): maximum speakers to predict
    """

    diarization_model_name: str = ""
    min_speakers: int = 1
    max_speakers: int = 2


class SinapsisWhisperxASRDiarization(SinapsisASRDiarization):
    """Template that runs an ASR with Diarization pipeline with Whisperx as ASR and Diarization engine

    Example agent:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: SinapsisWhisperxASRDiarization
    class_name: SinapsisWhisperxASRDiarization
    template_input: InputTemplate
    attributes:
        audio_file_path: null
        asr_model_name: '`replace_me:<class ''str''>`'
        diarization_model_name: ''
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0
        min_speakers: 1
        max_speakers: 2

    """

    AttributesBaseModel = WhisperxASRDiarizationAttributes

    def init_pipeline(self):
        return WhisperxASRDiarizationPipeline(
            asr_model_name=self.attributes.asr_model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
            min_speakers=self.attributes.min_speakers,
            max_speakers=self.attributes.max_speakers,
        )
