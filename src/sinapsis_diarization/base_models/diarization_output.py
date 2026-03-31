# -*- coding: utf-8 -*-
from dataclasses import dataclass
from datetime import timedelta
from re import compile

from numpy import ndarray
from pydantic import BaseModel, ConfigDict, Field
from sinapsis_core.data_containers.model_types._asr_base_models import (
    DiarizedTranscript,
    SpeechConstantKeys,
    TranscriptSegment,
)

SECONDS_PER_HOUR: int = 3600
SECONDS_PER_MINUTE: int = 60
MINUTES_PER_HOUR: int = 60
MILLISECONDS_PER_MINUTE: int = 1000


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16000
    max_secs: float = 30.0
    search_window: float = 5.0


class SpeechChunk(BaseModel):
    """Schema for raw audio segments from VAD."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")
    audio: ndarray | str = Field(..., description="The 16kHz audio buffer")

    @property
    def duration(self) -> float:
        return self.end - self.start


def format_timestamp(seconds: float) -> str:
    """Helper to convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // SECONDS_PER_HOUR
    minutes = (total_seconds % SECONDS_PER_HOUR) // MINUTES_PER_HOUR
    secs = total_seconds % SECONDS_PER_MINUTE
    millis = int(td.microseconds / MILLISECONDS_PER_MINUTE)
    return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"


def get_duration(segments) -> float:
    """Calculates total duration based on segments."""
    return segments[-1].end


def segments_to_text(segments) -> str:
    """Returns a formatted string of the conversation."""
    lines = []
    for s in segments:
        text = " ".join(s.text) if isinstance(s.text, list) else s.text
        lines.append(f"[{s.start:>6.2f}s - {s.end:>6.2f}s] {s.speaker}: {text}")
    return "\n".join(lines)


def segments_to_srt(segments) -> str:
    """Exports the transcript to SubRip (SRT) format."""
    srt_lines = []
    for i, s in enumerate(segments, 1):
        text = " ".join(s.text) if isinstance(s.text, list) else s.text
        srt_lines.append(f"{i}")
        srt_lines.append(f"{s.format_timestamp(s.start)} --> {s.format_timestamp(s.end)}")
        srt_lines.append(f"({s.speaker}) {text}\n")
    return "\n".join(srt_lines)


def segments_from_srt(srt_content: str) -> list[TranscriptSegment]:
    """Parses an SRT string back into a DiarizedTranscript object."""
    segments = []
    # Pattern for: Index \n Time --> Time \n (Speaker) Text
    pattern = compile(r"(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(?:\((.*?)\)\s)?(.*)")

    def to_sec(ts: str) -> float:
        h, m, s_ms = ts.split(":")
        s, ms = s_ms.split(",")
        return int(h) * SECONDS_PER_HOUR + int(m) * MINUTES_PER_HOUR + int(s) + int(ms) / MILLISECONDS_PER_MINUTE

    for match in pattern.finditer(srt_content):
        _, start_ts, end_ts, speaker, text = match.groups()
        segments.append(
            TranscriptSegment(
                start=to_sec(start_ts),
                end=to_sec(end_ts),
                text=text.strip(),
                speaker=speaker or SpeechConstantKeys.unknown_speaker_tag,
            )
        )
    return segments


class PredictedEmotion(BaseModel):
    name: str
    confidence: float | None = None


class EmotionTranscriptSegment(TranscriptSegment):
    emotions: list[PredictedEmotion]


class EmotionDiarizedTranscript(DiarizedTranscript):
    segments: list[EmotionTranscriptSegment]
