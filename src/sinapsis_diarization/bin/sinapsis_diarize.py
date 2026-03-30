# -*- coding: utf-8 -*-
from sinapsis_diarization.cli.arg_parser import args_parser
from sinapsis_diarization.cli.run_diarization import run_diarization


def main():
    args = args_parser()
    run_diarization(
        audio_path=args.audio,
        model=args.model,
        chunk_size=args.chunk_size_in_secs,
        model_name=args.model_name,
        device=args.device,
        sample_rate=args.sample_rate,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
