# -*- coding: utf-8 -*-
from abc import abstractmethod
from typing import Literal

from sinapsis_core.data_containers.annotations import AudioAnnotations
from sinapsis_core.data_containers.data_packet import DataContainer, TextPacket
from sinapsis_core.template_base import (
    Template,
    TemplateAttributes,
    TemplateAttributeType,
)

from sinapsis_diarization.base_models.diarization_output import AudioConfig


class BaseASRAttributes(TemplateAttributes):
    """
    model_name (str): name of model to run
    device (str): device to run the model
    sample_rate (int): sample rate of the audio
    chunk_size_in_secs (int): Size of the chunks to divide the audio. -1 for full audio
    search_window (float): Size of the search window
    audio_file_path (str): Path to audio file. Optional
    """

    model_name: str
    device: Literal["cpu", "cuda"] = "cuda"
    sample_rate: int = 16000
    chunk_size_in_secs: int = -1
    search_window: float = 5.0
    audio_file_path: str | None = None


class SinapsisBaseASR(Template):
    """Base template for ASR pipelines"""

    AttributesBaseModel = BaseASRAttributes

    def __init__(self, attributes: TemplateAttributeType):
        super().__init__(attributes)
        self.asr_pipeline = self.make_inference_engine()
        self.audio_config = AudioConfig(
            sample_rate=self.attributes.sample_rate,
            max_secs=self.attributes.chunk_size_in_secs,
            search_window=self.attributes.search_window,
        )

    @abstractmethod
    def make_inference_engine(self):
        pass

    def execute(self, container: DataContainer) -> DataContainer:
        """Call the ASR pipeline and add the information to container

        Args:
            container (DataContainer): data container

        Returns:
            DataContainer: data container with added information
        """
        if self.attributes.audio_file_path is not None:
            transcription_text: str = self.asr_pipeline.transcribe(
                audio_path_or_array=self.attributes.audio_file_path,
                config=self.audio_config,
            )
            text_packet = TextPacket(
                content=transcription_text,
                source=self.attributes.audio_file_path,
                generic_data={"produced_by_asr_model": self.attributes.model_name},
            )
            container.texts.append(text_packet)
        else:
            for audio_packet in container.audios:
                if audio_packet.sample_rate and audio_packet.sample_rate != self.attributes.sample_rate:
                    self.logger.warning("Sample rate in the incoming audio is not the same as used by the model")
                transcription_text = self.asr_pipeline.transcribe(
                    audio_path_or_array=audio_packet.content,
                    config=self.audio_config,
                )
                audio_packet.annotations.append(AudioAnnotations(raw_text=transcription_text))

        return container
