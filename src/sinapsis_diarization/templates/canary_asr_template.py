# -*- coding: utf-8 -*-
from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_CANARY_MODEL,
    SUPPORTED_CANARY_MODELS,
)
from sinapsis_diarization.pipelines.canary_asr import CanaryTranscriber
from sinapsis_diarization.templates.base_asr_template import (
    BaseASRAttributes,
    SinapsisBaseASR,
)


class SinapsisCanaryASR(SinapsisBaseASR):
    """Canary ASR template. Transcribes an audio using the Canary models

    Example agent:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: SinapsisCanaryASR
    class_name: SinapsisCanaryASR
    template_input: InputTemplate
    attributes:
        model_name: nvidia/canary-qwen-2.5b
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0
        audio_file_path: null


    """

    class AttributesBaseModel(BaseASRAttributes):
        model_name: SUPPORTED_CANARY_MODELS = DEFAULT_CANARY_MODEL

    def make_inference_engine(self):
        return CanaryTranscriber(
            model_name=self.attributes.model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
        )
