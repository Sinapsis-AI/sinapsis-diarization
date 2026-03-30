# -*- coding: utf-8 -*-

from sinapsis_diarization.base_models.supported_models import (
    DEFAULT_PYANNOTE_MODEL,
    SUPPORTED_PYANNOTE_MODELS,
)
from sinapsis_diarization.pipelines.pyannote_diarizer import PyannoteEngine
from sinapsis_diarization.templates.base_diarizer_template import (
    BaseDiarizerAttributes,
    SinapsisBaseDiarizer,
)


class PyannoteDiarizerAttributes(BaseDiarizerAttributes):
    """
    model_name (str): name of the model to run
    num_speakers (str): number of speakers to predict
    """

    model_name: SUPPORTED_PYANNOTE_MODELS = DEFAULT_PYANNOTE_MODEL
    num_speakers: int = 2


class SinapsisPyannoteDiarizer(SinapsisBaseDiarizer):
    """PyannoteDiarizer Template. Runs a diarization pipeline with pyannote models.

    Example agent:

    agent:
      name: my_test_agent
    templates:
    - template_name: InputTemplate
    class_name: InputTemplate
    attributes: {}
    - template_name: SinapsisPyannoteDiarizer
    class_name: SinapsisPyannoteDiarizer
    template_input: InputTemplate
    attributes:
        model_name: pyannote/speaker-diarization-community-1
        device: cuda
        sample_rate: 16000
        chunk_size_in_secs: -1
        search_window: 5.0
        audio_file_path: null
        num_speakers: 2

    """

    AttributesBaseModel = PyannoteDiarizerAttributes

    def make_inference_engine(self):
        return PyannoteEngine(
            model_name=self.attributes.model_name,
            device=self.attributes.device,
            sample_rate=self.attributes.sample_rate,
            chunk_size_in_secs=self.attributes.chunk_size_in_secs,
            num_speakers=self.attributes.num_speakers,
        )
