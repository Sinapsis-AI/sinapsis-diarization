# -*- coding: utf-8 -*-
from typing import Literal

import numpy as np
from sinapsis_core.data_containers._asr_base_models import DiarizedTranscript
from sinapsis_core.utils.logging_utils import sinapsis_logger

from sinapsis_diarization.base_models.diarization_output import (
    AudioConfig,
    EmotionDiarizedTranscript,
    EmotionTranscriptSegment,
    PredictedEmotion,
)
from sinapsis_diarization.base_models.supported_models import (
    SUPPORTED_PARAKEET_MODELS,
    SUPPORTED_PYANNOTE_MODELS,
    SUPPORTED_SORTFORMER_MODELS,
)
from sinapsis_diarization.pipelines.asr_diarization import (
    SUPPORTED_ASR_ENGINES,
    SUPPORTED_DIARIZER_ENGINES,
    ASRDiarizationPipeline,
)
from sinapsis_diarization.pipelines.base_processors import BaseEmotionEngine
from sinapsis_diarization.pipelines.context_manager import ManagedAudio
from sinapsis_diarization.pipelines.speechbrain_emotion_diarizer import (
    SpeechbrainEmotionEngine,
)

SUPPORTED_EMOTION_ENGINES = Literal["speechbrain"]


class ASRDiarizationEmotionPipeline(ASRDiarizationPipeline):
    """Pipeline that combines Automatic Speech Recognition, Diarization and Emotion Recognition"""

    def __init__(
        self,
        asr_pipeline: SUPPORTED_ASR_ENGINES,
        diarization_pipeline: SUPPORTED_DIARIZER_ENGINES,
        emotion_pipeline: SUPPORTED_EMOTION_ENGINES,
        asr_model_name: SUPPORTED_PARAKEET_MODELS,
        diarization_model_name: SUPPORTED_SORTFORMER_MODELS | SUPPORTED_PYANNOTE_MODELS,
        device: Literal["cpu", "cuda"] = "cuda",
        sample_rate: int = 16000,
        chunk_size_in_secs: int = -1,
        num_speakers: int = 2,
    ):
        super().__init__(
            asr_pipeline=asr_pipeline,
            diarization_pipeline=diarization_pipeline,
            asr_model_name=asr_model_name,
            diarization_model_name=diarization_model_name,
            device=device,
            sample_rate=sample_rate,
            chunk_size_in_secs=chunk_size_in_secs,
            num_speakers=num_speakers,
        )
        self.emotion_pipeline = self.init_emotion_pipeline(emotion_pipeline)

    def init_emotion_pipeline(self, emotion_pipeline: str) -> BaseEmotionEngine:
        """Chooses the emotion pipeline to initialize

        Args:
            emotion_pipeline (str): Name of the emotion pipeline

        Returns:
            BaseEmotionEngine: _description_
        """
        if emotion_pipeline == "speechbrain":
            return SpeechbrainEmotionEngine(
                device=self.device,
                sample_rate=self.sample_rate,
            )

    def transcribed_diarization(
        self, audio_path_or_array: str | np.ndarray, config: AudioConfig
    ) -> EmotionDiarizedTranscript:
        """Runs the ASR, Diarization and Emotion Recognition pipelines and aligns the results

        Args:
            audio_path_or_array (str | np.ndarray): path or array for audio
            config (AudioConfig): configuration for the audio

        Returns:
            EmotionDiarizedTranscript: Transcript of the audio with speaker and emotion labels
        """
        word_timestamps = []
        speakers_probs = []
        emotion_segments = []
        with ManagedAudio(audio_path_or_array, sample_rate=self.sample_rate, return_path=False) as audio_arr:
            grouped_segments = self.prepare_audio(audio_arr, config)
            if not grouped_segments:
                sinapsis_logger.warning("No segments to transcribe.")
                return ""

            for chunk in grouped_segments:
                sliced_audio = self.slice_audio(
                    audio=audio_arr,
                    start=int(chunk.start),
                    end=int(chunk.end),
                    sample_rate=self.sample_rate,
                    pad_samples=self.pad_samples,
                )
                word_timestamps.extend(self.asr_pipeline.get_asr_timestamps(audio_arr))
                with ManagedAudio(
                    sliced_audio,
                    sample_rate=self.sample_rate,
                    return_path=True,
                ) as audio_path:
                    segment_speaker_probs = self.diarization_pipeline.get_diarization_probs(audio_path)
                    speakers_probs.extend(segment_speaker_probs)
                    emotion_chunk_segments = self.emotion_pipeline._inference(audio_path)
                    emotion_segments.extend(emotion_chunk_segments)

        # 2. Tokenize into sentences
        sentences = self._tokenize_sentences(word_timestamps)

        # 3. Align and return
        script = self._align_sentences_to_speakers(sentences, word_timestamps, np.asarray(speakers_probs))
        return self._align_emotions_to_script(script, emotion_segments)

    def _align_emotions_to_script(
        self,
        script: DiarizedTranscript,
        emotion_segments: list[EmotionTranscriptSegment],
    ) -> EmotionDiarizedTranscript:
        """Aligns the emotions detected to the transcript

        Args:
            script (DiarizedTranscript): Transcript of audio
            emotion_segments (list[EmotionTranscriptSegment]): Segments of recognized emotions

        Returns:
            EmotionDiarizedTranscript: Aligned transcript with emotions.
        """
        final_script = EmotionDiarizedTranscript(filename=script.filename, segments=[])
        for segment in script.segments:
            emotions = self.find_segment_emotions(segment.start_time, segment.end_time, emotion_segments)
            new_segment = EmotionTranscriptSegment(
                text=segment.text,
                speaker=segment.speaker,
                start_time=segment.start_time,
                end_time=segment.end_time,
                emotions=emotions,
            )
            final_script.segments.append(new_segment)
        return final_script

    @staticmethod
    def find_segment_emotions(
        start: float, end: float, emotion_segments: list[EmotionTranscriptSegment]
    ) -> list[EmotionTranscriptSegment]:
        """Find emotions that appear in a given window of time

        Args:
            start (float): start of time segment
            end (float): end of time segment
            emotion_segments (list[EmotionTranscriptSegment]): list of emotions and their time
            appearances

        Returns:
            list[EmotionTranscriptSegment]: list of emotions and time segments
        """
        emotions: list[EmotionTranscriptSegment] = []
        for segment in emotion_segments:
            if segment.start_time >= start or segment.end_time <= end:
                segment_emotion = segment.emotions[0]
                emotion_names = {emotion.name for emotion in emotions}
                if segment_emotion.name not in emotion_names:
                    emotions.append(PredictedEmotion(name=segment_emotion.name))
        return emotions
