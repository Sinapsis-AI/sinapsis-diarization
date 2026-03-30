# -*- coding: utf-8 -*-
from typing import Literal

SUPPORTED_CANARY_MODELS = Literal["nvidia/canary-qwen-2.5b",]
DEFAULT_CANARY_MODEL: SUPPORTED_CANARY_MODELS = "nvidia/canary-qwen-2.5b"

SUPPORTED_PARAKEET_MODELS = Literal["nvidia/parakeet-tdt-0.6b-v2", "nvidia/parakeet-tdt-0.6b-v3"]
DEFAULT_PARAKEET_MODEL: SUPPORTED_PARAKEET_MODELS = "nvidia/parakeet-tdt-0.6b-v2"

SUPPORTED_SORTFORMER_MODELS = Literal["nvidia/diar_streaming_sortformer_4spk-v2.1",]
DEFAULT_SORTFORMER_MODEL: SUPPORTED_SORTFORMER_MODELS = "nvidia/diar_streaming_sortformer_4spk-v2.1"

SUPPORTED_PYANNOTE_MODELS = Literal["pyannote/speaker-diarization-community-1",]
DEFAULT_PYANNOTE_MODEL: SUPPORTED_PYANNOTE_MODELS = "pyannote/speaker-diarization-community-1"

DEFAULT_SPEECHBRAIN_MODEL = "speechbrain/emotion-diarization-wavlm-large"


SUPPORTED_WHISPERX_MODELS = Literal["tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v3"]
DEFAULT_WHISPERX_MODEL: SUPPORTED_WHISPERX_MODELS = "large-v3"
