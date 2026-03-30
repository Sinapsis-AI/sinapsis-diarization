# -*- coding: utf-8 -*-

from sinapsis_core.data_containers.annotations import AudioAnnotations
from sinapsis_core.data_containers.data_packet import AudioPacket, DataContainer

from sinapsis_diarization.base_models.diarization_output import (
    EmotionDiarizedTranscript,
)
from sinapsis_diarization.pipelines.asr_emotion_diarization import (
    ASRDiarizationEmotionPipeline,
)
from sinapsis_diarization.templates.asr_diarization_template import (
    SinapsisASRDiarization,
)


class SinapsisASREmotionDiarization(SinapsisASRDiarization):
    """Base template for ASR, diarization and emotion recognition pipelines"""

    def execute(self, container: DataContainer) -> DataContainer:
        if self.attributes.audio_file_path is not None:
            diarized_transcript: EmotionDiarizedTranscript = self.pipeline.transcribed_diarization(
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


class ParakeetPyannoteSpeechbrainASREmotionDiarization(SinapsisASREmotionDiarization):
    """Template that runs a pipeline for ASR with Diarization and Emotion Recognition
    using Parakeet as ASR, Pyannote as Diarization engine and Speechbrain as Emotion Engine.

    Example agent:

    agent:
    name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: ParakeetPyannoteSpeechbrainASREmotionDiarization
    class_name: ParakeetPyannoteSpeechbrainASREmotionDiarization
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
        return ASRDiarizationEmotionPipeline(
            asr_pipeline="parakeet",
            diarization_pipeline="pyannote",
            emotion_pipeline="speechbrain",
            asr_model_name=self.attributes.asr_model_name,
            diarization_model_name=self.attributes.diarization_model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
            num_speakers=self.attributes.num_speakers,
        )


class ParakeetSortformerSpeechbrainASREmotionDiarization(SinapsisASREmotionDiarization):
    """Template that runs a pipeline for ASR with Diarization and Emotion Recognition
    using Parakeet as ASR, Sortformer as Diarization engine and Speechbrain as Emotion Engine.

    Example agent:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: ParakeetSortformerSpeechbrainASREmotionDiarization
    class_name: ParakeetSortformerSpeechbrainASREmotionDiarization
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
        return ASRDiarizationEmotionPipeline(
            asr_pipeline="parakeet",
            diarization_pipeline="sortformer",
            emotion_pipeline="speechbrain",
            asr_model_name=self.attributes.asr_model_name,
            diarization_model_name=self.attributes.diarization_model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
        )
