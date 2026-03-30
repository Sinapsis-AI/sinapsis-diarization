# -*- coding: utf-8 -*-
from sinapsis_diarization.cli.arg_parser import asr_diarization_emotion_args_parser
from sinapsis_diarization.cli.run_asr_diarization_emotion import run_asr_diarization_emotion


def main():
    args = asr_diarization_emotion_args_parser()
    run_asr_diarization_emotion(
        audio_path=args.audio,
        asr_model=args.asr_model,
        diarization_model=args.diarization_model,
        emotion_model=args.emotion_model,
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
