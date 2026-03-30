[![sp](https://img.shields.io/badge/lang-sp-red.svg)](https://github.com/Sinapsis-AI/sinapsis-diarization/blob/main/README.es.md)
<h1 align="center">
<br>
<br>
<a href="https://sinapsis.tech/">
  <img
    src="https://github.com/Sinapsis-AI/brand-resources/blob/main/sinapsis_logo/4x/logo.png?raw=true"
    alt="" width="300">
</a>
<br>
Sinapsis Diarization
<br>
</h1>

<h4 align="center">Templates for Automatic Speech Recognition, Diarization and Emotion Recognition.</h4>

<p align="center">
<a href="#installation">🐍 Installation</a> •
<a href="#features">🚀 Features</a> •
<a href="#usage-example">📚 Example usage</a> •
<a href="#cli">CLI</a>
<a href="#documentation">📙 Documentation</a> •
<a href="#license">🔍 License</a>
</p>

The `sinapsis-diarization` module provides templates for real-time facial recognition with RetinaFace and DeepFace, enabling efficient and accurate inference.


<h2 id="installation">🐍 Installation</h2>

Install using your package manager of choice. We encourage the use of `uv`

```bash
uv pip install sinapsis-diarization
```
or wiht raw pip
```bash
pip install sinapsis-diarization
```

> [!IMPORTANT]
> Templates in sinapsis-diarization package may require extra dependencies. For development, we recommend installing the package with all the optional dependencies:

```bash
uv pip install sinapsis-diarization[all] --extra-index-url https://pypi.sinapsis.tech
```
or
```bash
pip install sinapsis-diarization[all] --extra-index-url https://pypi.sinapsis.tech
```


> [!IMPORTANT]
> Templates in sinapsis-diarization package may require a Huggingface Token. Set the environment variable for Hugginface using
> <code>export HF_TOKEN="your_huggingface_token"</code>



<h2 id="features">🚀 Features</h2>

<h3> Templates Supported</h3>

The **Sinapsis Diarization** module provides multiple templates for Automatic Speech Recognition, Diarization and Emotion Recognition.

- **SinapsisParakeetASR**: Runs Parakeet speech recognition for audio transcription.
- **SinapsisCanaryASR**: Runs Canary speech recognition for audio transcription
- **SinapsisSortformerDiarizer**: Runs Sortformer to get diarization of the speakers in an audio.
- **SinapsisPyannoteDiarizer**: Runs Pyannote to get diarization of speakers in an audio.
- **ParakeetPyannoteASRDiarization**: Runs Parakeet and Pyannote to transcribe an audio and divide it by speaker and time.
- **ParakeetSortformerASRDiarization**: Runs parakeet and Sortformer to transcribe an audio and divide it by speaker and time.
- **ParakeetPyannoteSpeechbrainASREmotionDiarization**: Runs Parakeet, Pyannote and Speechbrain to transcribe an audio, divide it by speaker and assign emotions to the segments.
- **ParakeetSortformerSpeechbrainASREmotionDiarization**: Runs Parakeet, Sortformer and Speechbrain to transcribe an audio, divide it by speaker and assign emotions to the segments.
- **SinapsisWhisperxASRDiarization**: Runs Whisperx to transcribe an audio and divide it by speaker and time.



<h2 id="usage-example">📚 Example usage</h2>

The following example demonstrates how to use the **SinapsisWhisperxASRDiarization** template for diarization.

This configuration defines an **agent** and a sequence of **templates** to run diarization with **Whisperx**. Provide an audio that you wish to transcribe at the audio_file_path attribute and choose one of the available models that fits your setup. *("tiny", "base", "small", "medium", "large-v1", "large-v2", "large-v2")*. Pick a maximum and minimum number of speakers. This config also allows for the audio to be divided in chunks of *n* seconds for easier processing.

<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Config file</span></strong></summary>

```yaml
agent:
  name: whisperx_asr_diarization
templates:
- template_name: InputTemplate
  class_name: InputTemplate
  attributes: {}
- template_name: SinapsisWhisperxASRDiarization
  class_name: SinapsisWhisperxASRDiarization
  template_input: InputTemplate
  attributes:
    audio_file_path: path to audio file
    asr_model_name: large-v3
    device: cuda
    sample_rate: 16000
    chunk_size_in_secs: -1
    min_speakers: 2
    max_speakers: 2
```
</details>

To run the agent, you should run:

```bash
sinapsis run /path/to/sinapsis-diarization/src/sinapsis_diarization/configs/transcription_with_diarization/whisperx_asr_diarization.yml
```


<h2 id="cli">📙 CLI</h2>

The pipelines for ASR, Diarization en Emotion diarization are available as CLI commands that take an audio and model options as input, and then transcribe/diarize the results in a given output directory (by default <code>results</code>):

<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Sinapsis ASR</span></strong></summary>

Run using <code>uv run sinapsis-asr</code>

Example:
<code>uv run sinapsis-asr --audio "path to audio" --model parakeet --chunk-size-in-secs 20 --device cuda
</code>

This command has the following options:

```bash
--audio AUDIO_PATH Path to audio
--model MODEL Type of model to run
--chunk-size-in-secs CHUNK_SIZE Size of chunks in seconds
--model-name MODEL_NAMEName of model to use
--device DEVICE Device to run the model
--sample-rate SAMPLE_RATE Sample rate of audio
--output-dir OUTPUT_DIR Output directory for transcription
```
**Models**
- "parakeet"
- "canary"

**Model names**
- Parakeet:
  - "nvidia/parakeet-tdt-0.6b-v2"
  - "nvidia/parakeet-tdt-0.6b-v3"

- Canary:
  - "nvidia/parakeet-tdt-0.6b-v2"

**Device options**
- "cuda"
- "cpu"


</details>

<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Sinapsis Diarize</span></strong></summary>

Run using <code>uv run sinapsis-diarize</code>

Example:
<code>uv run sinapsis-diarize --audio "path to audio" --model sortformer --chunk-size-in-secs 20 --device cuda</code>

This command has the following options:

```bash
--audio AUDIO_PATH Path to audio
--model MODEL Type of model to run
--chunk-size-in-secs CHUNK_SIZE Size of chunks in seconds
--model-name MODEL_NAME Name of model to use
--device DEVICE Device to run the model
--sample-rate SAMPLE_RATE Sample rate of audio
--output-dir OUTPUT_DIR Output directory for transcription
```
**Models**
- "sortformer"
- "pyannote"

**Model names**
- Sortformer:
  - "nvidia/diar_streaming_sortformer_4spk-v2.1"

- Pyannote:
  - "pyannote/speaker-diarization-community-1"

**Device options**
- "cuda"
- "cpu"


</details>

<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Sinapsis ASR Diarize</span></strong></summary>

Run using <code>uv run sinapsis-asr-diarize</code>

Example:

<code>uv run sinapsis-asr-diarize --audio "path to audio" --asr-model parakeet --diarization-model sortformer --chunk-size-in-secs 20 --device cuda</code>

This command has the following options:

```bash
--audio AUDIO_PATH Path to audio
--asr-model ASR_MODEL Type of ASR model to run
--diarization-model DIARIZATION_MODEL Type of Diarization model to run
--chunk-size-in-secs CHUNK_SIZE Size of chunks in seconds
--asr-model-name ASR_MODEL_NAME Name of ASR model to use
--diarization-model-name DIARIZATIN_MODEL_NAME Name of Diarization model to use
--device DEVICE Device to run the model
--sample-rate SAMPLE_RATE Sample rate of audio
--output-dir OUTPUT_DIR Output directory for transcription
--num-speakers NUM_SPEAKERS Number of speakers for models that require it

```
**ASR Models**
- "sortformer"
- "pyannote"

**Diarization Models**
- "sortformer"
- "pyannote"

**Model names**
- Parakeet:
  - "nvidia/parakeet-tdt-0.6b-v2"
  - "nvidia/parakeet-tdt-0.6b-v3"

- Canary:
  - "nvidia/parakeet-tdt-0.6b-v2"

- Sortformer:
  - "nvidia/diar_streaming_sortformer_4spk-v2.1"

- Pyannote:
  - "pyannote/speaker-diarization-community-1"

**Device options**
- "cuda"
- "cpu"


</details>

<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Sinapsis Whisperx ASR Diarize</span></strong></summary>

Run using <code>uv run sinapsis-whisperx-asr-diarize</code>

Example:

<code>uv run sinapsis-whisperx-asr-diarize --audio "path to audio" --model-name large-v3 --chunk-size-in-secs 20 --device cuda</code>

This command has the following options:

```bash
--audio AUDIO_PATH Path to audio
--model-name MODEL Type of ASR model to run
--chunk-size-in-secs CHUNK_SIZE Size of chunks in seconds
--device DEVICE Device to run the model
--sample-rate SAMPLE_RATE Sample rate of audio
--output-dir OUTPUT_DIR Output directory for transcription
--min-speakers MIN_SPEAKERS Minimum number of speakers
--max-speakers MAX_SPEAKERS Maximum number of speakers
```
**Model names**
- "tiny"
- "base"
- "small"
- "medium"
- "large-v1"
- "large-v2"
- "large-v3"

**Device options**
- "cuda"
- "cpu"


</details>

<details>
  <summary id="docker"><strong><span style="font-size: 1.2em;">Sinapsis ASR Diarize Emotion</span></strong></summary>

Run using <code>uv run sinapsis-asr-diarize-emotion</code>

Example:

<code>uv run sinapsis-asr-diarize-emotion --audio "path to audio" --asr-model parakeet --diarization-model sortformer --chunk-size-in-secs 20 --device cuda</code>

This command has the following options:

```bash
--audio AUDIO_PATH Path to audio
--asr-model ASR_MODEL Type of ASR model to run
--diarization-model DIARIZATION_MODEL Type of Diarization model to run
--emotion-model EMOTION_MODEL Type of Emotion model to run
--chunk-size-in-secs CHUNK_SIZE Size of chunks in seconds
--asr-model-name ASR_MODEL_NAME Name of ASR model to use
--diarization-model-name DIARIZATIN_MODEL_NAME Name of Diarization model to use
--device DEVICE Device to run the model
--sample-rate SAMPLE_RATE Sample rate of audio
--output-dir OUTPUT_DIR Output directory for transcription
--num-speakers NUM_SPEAKERS Number of speakers for models that require it

```
**ASR Models**
- "sortformer"
- "pyannote"

**Diarization Models**
- "sortformer"
- "pyannote"

**Emotion Models**
- "speechbrain"

**Model names**
- Parakeet:
  - "nvidia/parakeet-tdt-0.6b-v2"
  - "nvidia/parakeet-tdt-0.6b-v3"

- Canary:
  - "nvidia/parakeet-tdt-0.6b-v2"

- Sortformer:
  - "nvidia/diar_streaming_sortformer_4spk-v2.1"

- Pyannote:
  - "pyannote/speaker-diarization-community-1"

**Device options**
- "cuda"
- "cpu"


</details>


<h2 id="documentation">📙 Documentation</h2>

Documentation is available on the [sinapsis website](https://docs.sinapsis.tech/docs)

Tutorials for different projects within sinapsis are available at [sinapsis tutorials page](https://docs.sinapsis.tech/tutorials)

<h2 id="license">🔍 License</h2>

The templates in this project are licensed under the AGPLv3 license, which encourages open collaboration and sharing. For more details, please refer to the [LICENSE](LICENSE) file.

The command line interface and pipelines in this project are licensed under the MIT license, which allows for unrestricted use of the software and encourages open collaboration. For more details please refer to the [LICENSE](src/sinapsis_diarization/pipelines/LICENSE) file

For commercial use, please refer to our [official Sinapsis website](https://sinapsis.tech) for information on obtaining a commercial license.
