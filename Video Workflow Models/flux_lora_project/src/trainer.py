from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

from src.config import Config


def build_scheduler(optimizer, cfg: Config):
    name       = cfg.training.lr_scheduler.lower()
    total      = cfg.training.max_steps
    warmup     = cfg.training.lr_warmup_steps

    if name == "cosine":
        return get_cosine_schedule_with_warmup(optimizer,
                                               num_warmup_steps=warmup,
                                               num_training_steps=total)
    elif name == "linear":
        return get_linear_schedule_with_warmup(optimizer,
                                               num_warmup_steps=warmup,
                                               num_training_steps=total)
    elif name == "constant":
        from transformers import get_constant_schedule_with_warmup
        return get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup)
    else:
        raise ValueError(f"Unknown lr_scheduler: '{name}'. Choose: cosine | linear | constant")


def pack_latents(latents: torch.Tensor):
    """(B, C, H, W) → (B, H/2*W/2, C*4)"""
    B, C, H, W = latents.shape
    h2, w2     = H // 2, W // 2
    packed     = latents.reshape(B, C, h2, 2, w2, 2)
    packed     = packed.permute(0, 2, 4, 1, 3, 5)
    packed     = packed.reshape(B, h2 * w2, C * 4)
    return packed, h2, w2, C, H, W


def unpack_latents(packed: torch.Tensor, h2: int, w2: int, C: int, H: int, W: int, B: int):
    """(B, H/2*W/2, C*4) → (B, C, H, W)"""
    out = packed.reshape(B, h2, w2, C, 2, 2)
    out = out.permute(0, 3, 1, 4, 2, 5)
    out = out.reshape(B, C, H, W)
    return out


class FluxLoraTrainer:
    def __init__(self, cfg: Config, dataset: Dataset, output_dir: str, volume=None):
        self.cfg        = cfg
        self.dataset    = dataset
        self.output_dir = output_dir
        self.volume     = volume         
        self.device     = torch.device("cuda")

    def _load_pipeline(self):
        from diffusers import FluxPipeline

        print(f"Loading {self.cfg.model.name} ...")
        pipe = FluxPipeline.from_pretrained(
            self.cfg.model.name,
            revision=self.cfg.model.revision,
            torch_dtype=torch.bfloat16,
            token=self.cfg.credentials.hf_token,
        )
        pipe = pipe.to(self.device)
        return pipe

    def _add_lora(self, transformer):
        lc = self.cfg.lora
        lora_cfg = LoraConfig(
            r               = lc.rank,
            lora_alpha      = lc.alpha,
            target_modules  = lc.target_modules,
            lora_dropout    = lc.dropout,
            bias            = lc.bias,
        )
        transformer = get_peft_model(transformer, lora_cfg)
        transformer = transformer.to(torch.bfloat16).to(self.device)
        transformer.print_trainable_parameters()
        return transformer


    @torch.no_grad()
    def _encode_images(self, vae, pixel_values: torch.Tensor) -> torch.Tensor:
        latents = vae.encode(pixel_values).latent_dist.sample()
        latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        return latents.to(torch.bfloat16)

    @torch.no_grad()
    def _encode_text(self, tokenizer, text_encoder, tokenizer_2, text_encoder_2,
                     captions: list):
        clip_tokens = tokenizer(
            captions,
            padding="max_length",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        pooled = text_encoder(clip_tokens,
                               output_hidden_states=False).pooler_output.to(torch.bfloat16)

        t5_tokens = tokenizer_2(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(self.device)
        seq = text_encoder_2(t5_tokens,
                              output_hidden_states=False).last_hidden_state.to(torch.bfloat16)
        return pooled, seq


    def _training_step(self, transformer, vae, tokenizer, text_encoder,
                       tokenizer_2, text_encoder_2,
                       pixel_values: torch.Tensor, captions: list) -> torch.Tensor:
        bsz = pixel_values.shape[0]
        t   = self.cfg.training

        latents       = self._encode_images(vae, pixel_values)
        pooled, seq   = self._encode_text(tokenizer, text_encoder,
                                           tokenizer_2, text_encoder_2,
                                           list(captions))

        u         = torch.sigmoid(torch.randn((bsz,), device=self.device))
        timesteps = u * 1000.0
        alpha     = u.view(bsz, 1, 1, 1)

        noise         = torch.randn_like(latents)
        noisy_latents = (1.0 - alpha) * latents + alpha * noise

        packed, h2, w2, C, H, W = pack_latents(noisy_latents)

        img_ids         = torch.zeros(h2, w2, 3, device=self.device)
        img_ids[..., 1] = torch.arange(h2, device=self.device).unsqueeze(1)
        img_ids[..., 2] = torch.arange(w2, device=self.device).unsqueeze(0)
        img_ids         = img_ids.reshape(h2 * w2, 3)
        txt_ids         = torch.zeros(seq.shape[1], 3, device=self.device)

        guidance = torch.full((bsz,), t.guidance_scale,
                               device=self.device, dtype=torch.bfloat16)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred_packed = transformer(
                hidden_states       = packed,
                timestep            = timesteps / 1000.0,
                encoder_hidden_states = seq,
                pooled_projections  = pooled,
                txt_ids             = txt_ids,
                img_ids             = img_ids,
                guidance            = guidance,
                return_dict         = False,
            )[0]

        pred   = unpack_latents(pred_packed, h2, w2, C, H, W, bsz)
        target = noise - latents
        loss   = F.mse_loss(pred.float(), target.float())
        return loss


    def _save_checkpoint(self, transformer, step: int):
        ckpt_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
        transformer.save_pretrained(ckpt_dir)
        if self.volume:
            self.volume.commit()
        print(f"Checkpoint saved → {ckpt_dir}")


    def run(self):
        torch.cuda.init()
        torch.manual_seed(self.cfg.training.seed)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        from huggingface_hub import login
        login(token=self.cfg.credentials.hf_token)

        print(f"PyTorch : {torch.__version__}")
        print(f"CUDA   : {torch.cuda.get_device_name(0)}")

        pipe = self._load_pipeline()
        vae            = pipe.vae
        transformer    = pipe.transformer
        text_encoder   = pipe.text_encoder
        text_encoder_2 = pipe.text_encoder_2
        tokenizer      = pipe.tokenizer
        tokenizer_2    = pipe.tokenizer_2

        for model in [vae, text_encoder, text_encoder_2]:
            model.requires_grad_(False)
        transformer.requires_grad_(False)

        transformer = self._add_lora(transformer)

        t           = self.cfg.training
        dataloader  = DataLoader(
            self.dataset,
            batch_size  = t.batch_size,
            shuffle     = True,
            num_workers = t.num_workers,
            drop_last   = True,
        )

        optimizer = torch.optim.AdamW(
            transformer.parameters(),
            lr           = t.learning_rate,
            betas        = (0.9, 0.999),
            weight_decay = 0.01,
            eps          = 1e-8,
        )
        scheduler = build_scheduler(optimizer, self.cfg)

        print(f"\nTraining started")
        print(f"   Steps        : {t.max_steps}")
        print(f"   Batch size   : {t.batch_size} × {t.gradient_accum_steps} = "
              f"{t.batch_size * t.gradient_accum_steps} effective")
        print(f"   LR           : {t.learning_rate}  ({t.lr_scheduler})")
        print(f"   LoRA rank    : {self.cfg.lora.rank}")
        print(f"   Output       : {self.output_dir}\n")

        transformer.train()
        global_step = 0
        accum_loss  = 0.0
        optimizer.zero_grad()
        data_iter   = iter(dataloader)
        pbar        = tqdm(total=t.max_steps, desc="Steps")

        while global_step < t.max_steps:
            try:
                pixel_values, captions = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                pixel_values, captions = next(data_iter)

            pixel_values = pixel_values.to(self.device, dtype=torch.bfloat16)

            loss = self._training_step(
                transformer, vae, tokenizer, text_encoder,
                tokenizer_2, text_encoder_2, pixel_values, captions
            )
            loss = loss / t.gradient_accum_steps
            loss.backward()
            accum_loss += loss.item()

            if (global_step + 1) % t.gradient_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), t.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                pbar.set_postfix(
                    loss=f"{accum_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}"
                )
                accum_loss = 0.0

            global_step += 1
            pbar.update(1)

            if global_step % self.cfg.checkpointing.save_steps == 0:
                self._save_checkpoint(transformer, global_step)

        pbar.close()

        print("\nSaving final model ...")
        transformer.save_pretrained(self.output_dir)
        if self.volume:
            self.volume.commit()

        print(f"\nTraining complete!  Model saved to: {self.output_dir}")
        return self.output_dir
