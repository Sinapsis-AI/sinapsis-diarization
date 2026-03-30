# -*- coding: utf-8 -*-
import os
import tempfile
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from sinapsis_core.utils.logging_utils import sinapsis_logger


class ManagedAudio:
    def __init__(
        self,
        data: np.ndarray | str | Path,
        sample_rate: int,
        target_sr: int | None = None,
        return_path: bool = True,  # The flag that controls __enter__
    ):
        self.data = data
        self.src_sr = sample_rate
        self.target_sr = target_sr or sample_rate
        self.return_path = return_path
        self.temp_path: str | None = None

    def _get_processed_array(self) -> np.ndarray:
        """Internal helper to load/resample the data into an array."""
        if isinstance(self.data, np.ndarray):
            if self.target_sr != self.src_sr:
                return librosa.resample(self.data.astype(np.float32), orig_sr=self.src_sr, target_sr=self.target_sr)
            return self.data

        # Load from path and resample to target_sr
        arr, _ = librosa.load(self.data, sr=self.target_sr)
        return arr

    def __enter__(self) -> str | Path | np.ndarray:
        # Case 1: User wants a Numpy Array
        if not self.return_path:
            return self._get_processed_array()

        # Case 2: User wants a File Path
        # Optimization: Input is already a path and rates match
        if isinstance(self.data, (str, Path)) and self.src_sr == self.target_sr:
            return self.data

        # Otherwise, create a temp file
        fd, self.temp_path = tempfile.mkstemp(suffix=".wav")
        with os.fdopen(fd, "wb") as f:
            sf.write(f, self._get_processed_array(), self.target_sr, format="WAV")

        return self.temp_path

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.temp_path and os.path.exists(self.temp_path):
            try:
                os.unlink(self.temp_path)
            except OSError:
                sinapsis_logger.error(f"Failed to remove temporary file {self.temp_path}")
