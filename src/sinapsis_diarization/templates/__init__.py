# -*- coding: utf-8 -*-
import importlib
from typing import Any, Callable, cast

_root_lib_path = "sinapsis_diarization.templates"

_template_lookup = {
    "SinapsisParakeetASR": f"{_root_lib_path}.parakeet_asr_template",
    "SinapsisCanaryASR": f"{_root_lib_path}.canary_asr_template",
    "SinapsisSortformerDiarizer": f"{_root_lib_path}.sortformer_diarizer_template",
    "SinapsisPyannoteDiarizer": f"{_root_lib_path}.pyannote_diarizer_template",
    "ParakeetPyannoteASRDiarization": f"{_root_lib_path}.asr_diarization_template",
    "ParakeetSortformerASRDiarization": f"{_root_lib_path}.asr_diarization_template",
    "ParakeetPyannoteSpeechbrainASREmotionDiarization": f"{_root_lib_path}.asr_emotion_diarization_template",
    "ParakeetSortformerSpeechbrainASREmotionDiarization": f"{_root_lib_path}.asr_emotion_diarization_template",
    "SinapsisWhisperxASRDiarization": f"{_root_lib_path}.whisperx_asr_diarization",
}


def __getattr__(name: str) -> Callable[..., Any]:
    if name in _template_lookup:
        module = importlib.import_module(_template_lookup[name])
        attr = getattr(module, name)
        if callable(attr):
            return cast(Callable[..., Any], attr)
        raise TypeError(f"Attribute `{name}` in `{_template_lookup[name]}` is not callable.")

    raise AttributeError(f"template `{name}` not found in {_root_lib_path}")


__all__ = list(_template_lookup.keys())
