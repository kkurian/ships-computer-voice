# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified from finetuning/sft_12hz.py of https://github.com/QwenLM/Qwen3-TTS at
# commit 022e286b98fbec7e1e916cb940cdf532cd9f488e, by Kerry Ivan Kurian, 2026.
# Every change is marked "PATCH" and says why: label alignment (upstream issues
# #371 and #337, PR #278), the text projection (PR #336), sdpa attention, fp32
# master weights, bf16 export, an atomic checkpoint write (issue #365), a SAVED
# marker per epoch, and loss logging. This file stays under the Apache License
# 2.0 (see LICENSE-APACHE-2.0 in this repository), not MIT.
import argparse
import json
import os
import shutil
import time

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None

def sub_talker_loss_aligned(talker, codec_ids, talker_hidden_states):
    # PATCH (issue #371 / PR #278): same inputs as forward_sub_talker_finetune, but the labels
    # (codes 1..15) are already aligned with the 15 logits, so no extra causal shift.
    embs = [talker_hidden_states.unsqueeze(1), talker.get_input_embeddings()(codec_ids[:, :1])]
    embs += [talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, i:i + 1]) for i in range(1, talker.config.num_code_groups - 1)]
    logits = talker.code_predictor.forward_finetune(inputs_embeds=torch.cat(embs, dim=1)).logits
    return torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), codec_ids[:, 1:].reshape(-1))
def train():
    global target_speaker_embedding

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.float32,   # PATCH: fp32 master weights; bf16 AdamW rounds away updates below bf16 resolution
        attn_implementation="sdpa",  # PATCH: no flash-attn build; same maths
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    t_start = time.time()
    for epoch in range(num_epochs):
        ep_losses = []
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']

                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids)
                input_text_embedding = model.talker.text_projection(input_text_embedding)  # PATCH (PR #336)
                input_text_embedding = input_text_embedding * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # PATCH (issue #371 / PR #278): HF ForCausalLMLoss shifts labels itself; pass them unshifted.
                outputs = model.talker(
                    inputs_embeds=input_embeddings,
                    attention_mask=attention_mask,
                    labels=codec_0_labels,
                    output_hidden_states=True
                )

                # PATCH (issue #337 / PR #278): the sub-talker for frame t reads the hidden state at t-1
                # (the position that predicted code0 of frame t), as at inference; not the leaked one at t.
                hidden_states = outputs.hidden_states[0][-1][:, :-1, :]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_loss = sub_talker_loss_aligned(model.talker, talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            ep_losses.append(loss.item())
            if epoch == 0 and step < 3:
                accelerator.print(f"step {step}: talker CE {outputs.loss.item():.4f} | sub-talker CE {sub_talker_loss.item():.4f}", flush=True)
            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f} | {time.time()-t_start:.0f}s", flush=True)

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            config_dict["tts_model_type"] = "custom_voice"
            talker_config = config_dict.get("talker_config", {})
            talker_config["spk_id"] = {
                args.speaker_name: 3000
            }
            talker_config["spk_is_dialect"] = {
                args.speaker_name: False
            }
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu").to(torch.bfloat16) for k, v in unwrapped_model.state_dict().items()}  # PATCH: export bf16

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            state_dict['talker.model.codec_embedding.weight'][3000] = target_speaker_embedding[0].detach().to(weight.device).to(weight.dtype)
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path + ".tmp")   # PATCH: atomic (issue #365)
            os.replace(save_path + ".tmp", save_path)
            del state_dict
            with open(os.path.join(output_dir, "SAVED"), "w") as f:
                json.dump({"epoch": epoch, "steps": len(ep_losses), "mean_loss": sum(ep_losses)/max(1,len(ep_losses)),
                           "last_loss": ep_losses[-1] if ep_losses else None, "elapsed_s": time.time()-t_start}, f)
            accelerator.print(f"SAVED epoch {epoch} -> {output_dir}", flush=True)

if __name__ == "__main__":
    train()
