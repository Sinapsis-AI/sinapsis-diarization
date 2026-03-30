# -*- coding: utf-8 -*-
from sinapsis_diarization.cli.arg_parser import whisperx_asr_diarization_args_parser
from sinapsis_diarization.cli.run_asr_diarization import run_whisperx_asr_diarization


def main():
    args = whisperx_asr_diarization_args_parser()
    run_whisperx_asr_diarization(
        audio_path=args.audio,
        model_name=args.model_name,
        chunk_size=args.chunk_size_in_secs,
        device=args.device,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )


if __name__ == "__main__":
    main()
