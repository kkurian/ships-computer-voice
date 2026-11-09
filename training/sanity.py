#!/usr/bin/env python3
"""Render two lines with each saved checkpoint on the GPU, to catch a broken run early.

    python3 sanity.py OUTPUT_DIR SPEAKER

Writes OUTPUT_DIR/checkpoint-epoch-N/sanity-lineK.wav and OUTPUT_DIR/sanity.json
with each render's length and time. A healthy checkpoint speaks each line in a
few seconds; a broken one babbles until the token cap (about 48 s of audio),
which is what the failed first run did.
"""
import glob
import json
import os
import sys
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

LINES = {
    3: "Warning. Test suite failed. Two examples failing in the build queue.",
    7: "Status report. The build queue processed four changes. Three passed their tests and "
       "were merged. One failed a regression check and was returned to the author.",
}
CAP_S = 47  # 600 tokens at 12.5 per second is 48 s


def main() -> int:
    out, speaker = sys.argv[1], sys.argv[2]
    result = {}
    for d in sorted(glob.glob(f"{out}/checkpoint-epoch-*")):
        if not os.path.exists(f"{d}/SAVED"):
            continue
        tts = Qwen3TTSModel.from_pretrained(d, device_map="cuda:0", dtype=torch.bfloat16,
                                            attn_implementation="sdpa")
        r = {}
        for k, text in LINES.items():
            t0 = time.time()
            wavs, sr = tts.generate_custom_voice(text=text, speaker=speaker, language="Auto",
                                                 max_new_tokens=600)
            sf.write(f"{d}/sanity-line{k}.wav", wavs[0], sr)
            secs = len(wavs[0]) / sr
            r[k] = {"audio_s": round(secs, 2), "gen_s": round(time.time() - t0, 2), "hit_cap": secs >= CAP_S}
        result[os.path.basename(d)] = r
        print(os.path.basename(d), r, flush=True)
        del tts
        torch.cuda.empty_cache()
    with open(f"{out}/sanity.json", "w") as f:
        json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
