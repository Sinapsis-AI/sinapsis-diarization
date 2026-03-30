# -*- coding: utf-8 -*-

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_SORTFORMER_MODEL,
    SUPPORTED_SORTFORMER_MODELS,
)
from sinapsis_diarization.pipelines.sortformer_diarizer import SortformerEngine
from sinapsis_diarization.templates.base_diarizer_template import (
    BaseDiarizerAttributes,
    SinapsisBaseDiarizer,
)


class SortformerDiarizerAttributes(BaseDiarizerAttributes):
    model_name: SUPPORTED_SORTFORMER_MODELS = DEFAULT_SORTFORMER_MODEL


class SinapsisSortformerDiarizer(SinapsisBaseDiarizer):
    """Sortformer Diarizer Template. Runs a diarization pipeline with sortformer models.

    Example agent:

    agent:
        name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: SinapsisSortformerDiarizer
    class_name: SinapsisSortformerDiarizer
    template_input: InputTemplate
    attributes:
        model_name: nvidia/diar_streaming_sortformer_4spk-v2.1
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0
        audio_file_path: null


    """

    AttributesBaseModel = SortformerDiarizerAttributes

    def make_inference_engine(self):
        return SortformerEngine(
            model_name=self.attributes.model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
        )
