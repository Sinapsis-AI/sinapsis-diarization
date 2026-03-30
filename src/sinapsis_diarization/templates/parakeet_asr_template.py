# -*- coding: utf-8 -*-
from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_PARAKEET_MODEL,
    SUPPORTED_PARAKEET_MODELS,
)
from sinapsis_diarization.pipelines.parakeet_asr import ParakeetASR
from sinapsis_diarization.templates.base_asr_template import (
    BaseASRAttributes,
    SinapsisBaseASR,
)


class SinapsisParakeetASR(SinapsisBaseASR):
    """Parakeet ASR template. Transcribes an audio using the Parakeet models

    Example agent:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: SinapsisParakeetASR
    class_name: SinapsisParakeetASR
    template_input: InputTemplate
    attributes:
        model_name: nvidia/parakeet-tdt-0.6b-v2
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0
        audio_file_path: null

    """

    class AttributesBaseModel(BaseASRAttributes):
        model_name: SUPPORTED_PARAKEET_MODELS = DEFAULT_PARAKEET_MODEL

    def make_inference_engine(self):
        return ParakeetASR(
            model_name=self.attributes.model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
        )
