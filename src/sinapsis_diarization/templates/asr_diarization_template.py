# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Literal

from sinapsis_core.data_containers._asr_base_models import DiarizedTranscript
from sinapsis_core.data_containers.annotations import AudioAnnotations
from sinapsis_core.data_containers.data_packet import AudioPacket, DataContainer
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)

from sinapsis_diarization.base_models.diarization_output import AudioConfig
from sinapsis_diarization.pipelines.asr_diarization import (
    ASRDiarizationPipeline,
)


class ASRDiarizationAttributes(TemplateAttributes):
    """
    audio_file_path(str): Path to audio file. Optional
    asr_model_name (str): Name of the ASR pipeline to use
    diarization_model_name (str): Name of the diarization pipeline
    device (str): device to run the model
    sample_rate (int): sample rate of the audio
    chunk_size_in_secs (int): Size of the chunks to divide the audio. -1 for full audio
    search_window (float): Size of the search window
    """

    audio_file_path: str | None = None
    asr_model_name: str
    diarization_model_name: str
    device: Literal["cpu", "cuda"] = "cuda"
    sample_rate: int = 16000
    chunk_size_in_secs: int = -1
    search_window: float = 5.0


class SinapsisASRDiarization(Template):
    """Base template for ASR Diarization pipelines"""

    AttributesBaseModel = ASRDiarizationAttributes

    def __init__(self, attributes: TemplateAttributeType):
        super().__init__(attributes)
        self.pipeline = self.init_pipeline()
        self.audio_config = AudioConfig(
            sample_rate=self.attributes.sample_rate,
            max_secs=self.attributes.chunk_size_in_secs,
            search_window=self.attributes.search_window,
        )

    @abstractmethod
    def init_pipeline(self):
        pass

    def execute(self, container: DataContainer) -> DataContainer:
        """Call the ASR Diarization pipeline and add the information to container

        Args:
            container (DataContainer): data container

        Returns:
            DataContainer: data container with added information
        """
        if self.attributes.audio_file_path is not None:
            diarized_transcript: DiarizedTranscript = self.pipeline.transcribed_diarization(
                audio_path_or_array=self.attributes.audio_file_path,
                config=self.audio_config,
            )
            audio_packet = AudioPacket(content="", source=self.attributes.audio_file_path)
            annotations = AudioAnnotations(diarized_transcript=diarized_transcript)
            audio_packet.annotations = annotations
            container.audios.append(audio_packet)
        else:
            for audio_packet in container.audios:
                if audio_packet.sample_rate and audio_packet.sample_rate != self.attributes.sample_rate:
                    self.logger.warning("Sample rate in the incoming audio is not the same as used by the model")
                diarized_transcript = self.pipeline.transcribed_diarization(
                    audio_path_or_array=audio_packet.content,
                    config=self.audio_config,
                )
                audio_packet.annotations = AudioAnnotations(diarized_transcript=diarized_transcript)

        return container


class PyannoteAttributes(ASRDiarizationAttributes):
    """num_speakers (int): Number of speakers to predict"""

    num_speakers: int = 2


class ParakeetPyannoteASRDiarization(SinapsisASRDiarization):
    """Template that runs an ASR with Diarization pipeline with Parakeet as ASR engine
    and Pyannote as Diarization engine

    Example agent:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: ParakeetPyannoteASRDiarization
    class_name: ParakeetPyannoteASRDiarization
    template_input: InputTemplate
    attributes:
        audio_file_path: null
        asr_model_name: '`replace_me:<class ''str''>`'
        diarization_model_name: '`replace_me:<class ''str''>`'
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0
        num_speakers: 2


    """

    AttributesBaseModel = PyannoteAttributes

    def init_pipeline(self):
        return ASRDiarizationPipeline(
            asr_pipeline="parakeet",
            diarization_pipeline="pyannote",
            asr_model_name=self.attributes.asr_model_name,
            diarization_model_name=self.attributes.diarization_model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
            num_speakers=self.attributes.num_speakers,
        )


class ParakeetSortformerASRDiarization(SinapsisASRDiarization):
    """Template that runs an ASR with Diarization pipeline with Parakeet as ASR engine
    and Sortformer as Diarization engine

    Example agent:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: ParakeetSortformerASRDiarization
    class_name: ParakeetSortformerASRDiarization
    template_input: InputTemplate
    attributes:
        audio_file_path: null
        asr_model_name: '`replace_me:<class ''str''>`'
        diarization_model_name: '`replace_me:<class ''str''>`'
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0

    """

    def init_pipeline(self):
        return ASRDiarizationPipeline(
            asr_pipeline="parakeet",
            diarization_pipeline="sortformer",
            asr_model_name=self.attributes.asr_model_name,
            diarization_model_name=self.attributes.diarization_model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
        )
