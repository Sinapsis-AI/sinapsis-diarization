# -*- coding: utf-8 -*-
from sinapsis_diarization.cli.arg_parser import asr_diarization_args_parser
from sinapsis_diarization.cli.run_asr_diarization import run_asr_diarization


def main():
    args = asr_diarization_args_parser()
    run_asr_diarization(
        audio_path=args.audio,
        asr_model=args.asr_model,
        diarization_model=args.diarization_model,
        chunk_size=args.chunk_size_in_secs,
        asr_model_name=args.asr_model_name,
        diarization_model_name=args.diarization_model_name,
        device=args.device,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
        num_speakers=args.num_speakers,
    )


if __name__ == "__main__":
    main()
