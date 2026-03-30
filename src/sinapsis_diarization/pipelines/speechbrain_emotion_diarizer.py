# -*- coding: utf-8 -*-
from os.path import join
from typing import Literal

from sinapsis_core.utils.env_var_keys import SINAPSIS_CACHE_DIR
from speechbrain.inference.diarization import Speech_Emotion_Diarization

from sinapsis_diarization.base_models.diarization_output import (
    EmotionTranscriptSegment,
    PredictedEmotion,
)
from sinapsis_diarization.base_models.supported_models import DEFAULT_SPEECHBRAIN_MODEL
from sinapsis_diarization.pipelines.base_processors import BaseEmotionEngine


class SpeechbrainEmotionEngine(BaseEmotionEngine):
    """Pipeline for Speechbrain Emotion Recognition"""

    def __init__(
        self,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        model_save_dir: str = join(SINAPSIS_CACHE_DIR, "pretrained_models/emotion_diarizer"),
    ):
        self.model_save_dir = model_save_dir
        super().__init__(
            device=device,
            sample_rate=sample_rate,
        )

    def _load_model_weights(self):
        model = Speech_Emotion_Diarization.from_hparams(
            source=DEFAULT_SPEECHBRAIN_MODEL,
            savedir=self.model_save_dir,
            run_opts={"device": self.device},
        )
        return model

    def _inference(self, audio_path: str) -> list[EmotionTranscriptSegment]:
        """Gets emotion by segments from an audio

        Args:
            audio_path (str): path to audio

        Returns:
            list[EmotionTranscriptSegment]: list of emotion segments
        """
        diarization_result: dict[str, list[dict]] = self.model.diarize_file(audio_path)

        segments = []
        for segment in diarization_result[audio_path]:
            emotion = segment["emotion"]

            # Mapping labels: 'n' -> neutral, 'h' -> happy, 's' -> sad, 'a' -> angry
            label_map = {"n": "Neutral", "h": "Happy", "s": "Sad", "a": "Angry"}
            emotion_full = label_map.get(emotion, emotion)
            segments.append(
                EmotionTranscriptSegment(
                    text="",
                    start_time=segment["start"],
                    end_time=segment["end"],
                    emotions=[PredictedEmotion(name=emotion_full)],
                )
            )

        return segments
