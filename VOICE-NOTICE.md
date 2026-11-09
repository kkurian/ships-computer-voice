# Voice notice

This notice covers the voice weights published as assets of the v2.0.0 release: `ships-computer-v1-mlx-6bit.tar` and `ships-computer-v1-hf-bf16.tar`, and everything in them. It does not cover the code and documentation, which are MIT (see [LICENSE](LICENSE)), except `training/sft_12hz_patched.py`, which is Apache-2.0.

- The voice ("version 1") is a fine-tune of Qwen's [Qwen3-TTS-12Hz-0.6B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base). That base model is licensed under the Apache License 2.0 (Copyright the Alibaba Qwen team); a copy is in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and in each release archive. The weights were changed by fine-tuning, as the README describes. The speech tokenizer in the archives is Qwen's, unchanged.
- It was fine-tuned on 425 transcribed clips (21 minutes) of the Star Trek: The Next Generation computer voice, performed by Majel Barrett, selected from the dataset [STCOMP/majel_tts_medium_clean](https://huggingface.co/datasets/STCOMP/majel_tts_medium_clean). That dataset states no licence. The clips, their transcripts and the reference clip are not published here.
- Rights in those recordings belong to their owners. We hold none, so we do not license the fine-tuned weights under MIT, Apache-2.0 or any other licence, beyond what Apache-2.0 already grants for Qwen's own work in them.
- This is an unofficial fan project. It is not affiliated with or endorsed by Paramount, CBS, the Roddenberry estate or the Barrett estate. The original performer's estate was not consulted.
- The weights are provided as-is for personal, non-commercial use. Do not use them to deceive anyone into thinking a real person said something, and do not sell them or use them in a paid product or service.
- The earlier voice of release v1.0.0 keeps the notice published with it.
