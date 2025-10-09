"""
w2vbert 2.0 continue pretrain code on kaggle
"""
from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
    FullyStateDictConfig,
    StateDictType,
    ShardingStrategy,
)

# -------------------------------
# Config dataclass
# -------------------------------
@dataclass
class TrainConfig:
    manifest: str
    output_dir: str = "./checkpoints"
    batch_size: int = 8
    num_workers: int = 4
    max_epochs: int = 10
    total_steps: int = 20000
    warm_head_steps: int = 2000
    encoder_lr: float = 3e-5
    head_lr: float = 5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    accumulate_steps: int = 1
    seed: int = 42
    device: str = "cuda"
    vocab_size: int = 4096
    hidden_size: int = 768
    save_every_steps: int = 2000
    log_every_steps: int = 50
    resume_from: Optional[str] = None
    use_mixed_precision: bool = True
    shuffle_manifest: bool = True
    curriculum_enabled: bool = True 
    min_chunk_sec: float = 3.0
    max_chunk_sec: float = 30.0
    within_utt_neg_ratio: float = 0.7  # mix within-utterance and across-utt negatives


# -------------------------------
# Utility helpers
# -------------------------------


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


# -------------------------------
# Dataset & Collate
# -------------------------------
class ViBaSpeechAudioDataset(Dataset):

    def __init__(self, ViBa_audio_path: str, sample_rate: int = 16000):
        with open(ViBa_audio_path, "r", encoding="utf-8") as fh:
            lines = [json.loads(l) for l in fh if l.strip()]
        self.items = lines
        self.sample_rate = sample_rate
    def __len__(self) -> int:
        return len(self.items)

    def _load_wave(self, path: str) -> Tensor:
        # torchaudio returns (channels, samples)
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return waveform.squeeze()

    def _sample_chunk(self, waveform: Tensor, duration_sec: float) -> Tensor:
        num_samples = waveform.size(1)
        target_samples = int(duration_sec * self.sample_rate)
        if target_samples >= num_samples:
            # pad if too short
            pad = target_samples - num_samples
            return F.pad(waveform, (0, pad))
        # random crop
        start = random.randint(0, num_samples - target_samples)
        return waveform[:, start:start + target_samples]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.items[idx]
        path = item["path"]
        # load
        waveform = self._load_wave(path)
        return {
            "waveform": waveform,  # (samples,)
            "orig_path": path,
            "duration": waveform.size(0) / self.sample_rate,
        }


def collate_audio_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    # Pad waveforms to max length in batch
    max_len = max(x["waveform"].size(0) for x in batch)
    waveforms = []
    lengths = []
    speakers = []
    utt_ids = []
    for x in batch:
        w = x["waveform"]
        pad = max_len - w.size(1)
        if pad > 0:
            w = F.pad(w, (0, pad))
        waveforms.append(w)
        lengths.append(x["waveform"].size(1))
        speakers.append(x["speaker"])
        utt_ids.append(x["utt_id"])
    waveforms = torch.cat(waveforms, dim=0)  # (B, T) if mono
    return {
        "waveforms": waveforms,  # shape (B, T)
        "lengths": torch.tensor(lengths, dtype=torch.int64),
        "speakers": speakers,
        "utt_ids": utt_ids,
    }


# -------------------------------
# Model wrapper: encoder + mlm head
# -------------------------------
class MLMHead(nn.Module):
    def __init__(self, hidden_dim: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, vocab_size)
        # initialization similar to HuggingFace defaults
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, features: Tensor) -> Tensor:
        # features: (B, T, H) -> logits (B, T, V)
        return self.proj(features)


# -------------------------------
# Training loop class
# -------------------------------
class Trainer:
    def __init__(self, config: TrainConfig, model: W2VBertForContinuePretrain, train_loader: DataLoader, val_loader: Optional[DataLoader] = None):
        self.cfg = config
        self.device = torch.device(self.cfg.device if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        # optimizer with param groups (head vs encoder)
        self.optimizer = AdamW([
            {"params": self.model.mlm_head.parameters(), "lr": self.cfg.head_lr},
            {"params": self.model.encoder.parameters(), "lr": self.cfg.encoder_lr}
        ], weight_decay=self.cfg.weight_decay)

        self.scaler = GradScaler(enabled=self.cfg.use_mixed_precision)
        self.global_step = 0
        self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, "tb_logs"))
        ensure_dir(self.cfg.output_dir)

        # checkpoint state
        self.best_val_loss = float("inf")

        # freeze encoder initially for warm head stage
        self._freeze_encoder()

    def _freeze_encoder(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = False

    def _unfreeze_encoder(self) -> None:
        for p in self.model.encoder.parameters():
            p.requires_grad = True

    def save_checkpoint(self, name: str = None) -> str:
        if name is None:
            name = f"ckpt_step_{self.global_step}.pt"
        out_path = os.path.join(self.cfg.output_dir, name)
        state = {
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optim_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "cfg": vars(self.cfg),
        }
        torch.save(state, out_path)
        return out_path

    def load_checkpoint(self, path: str, strict: bool = False) -> None:
        state = torch.load(path, map_location="cpu")
        self.model.load_state_dict(state["model_state"], strict=strict)
        self.optimizer.load_state_dict(state["optim_state"]) if "optim_state" in state else None
        if "scaler_state" in state and self.cfg.use_mixed_precision:
            self.scaler.load_state_dict(state["scaler_state"])
        self.global_step = state.get("global_step", 0)
        print(f"Resumed from {path} at step {self.global_step}")

    def compute_mlm_loss(self, logits: Tensor, targets: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        # logits: (B, T, V), targets: (B, T) with values in [0, V)
        b, t, v = logits.shape
        logits = logits.view(b * t, v)
        targets = targets.view(b * t)
        loss = F.cross_entropy(logits, targets, reduction="none")
        if mask is not None:
            loss = loss * mask.view(-1).float()
            denom = mask.sum().clamp(min=1.0)
            return loss.sum() / denom
        return loss.mean()

    def train_step(self, batch: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        waveforms = batch["waveforms"].to(self.device)
        lengths = batch["lengths"].to(self.device) if "lengths" in batch else None

        # For the MLM targets you must implement target creation logic.
        # Here we leave a hook: encode_targets_with_quantizer(waveforms)
        # which should return `targets` (B, T_idx) aligning with logits output.

        with autocast(enabled=self.cfg.use_mixed_precision):
            out = self.model(waveforms, lengths=lengths)
            logits = out["logits"]  # (B, T, V)

            # NOTE: placeholder target creation. Replace with real target creation.
            targets, mask = self._create_placeholder_targets(logits)

            loss = self.compute_mlm_loss(logits, targets, mask)

        self.scaler.scale(loss).backward()
        # gradient accumulation handled externally if needed
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        metrics = {"mlm_loss": float(loss.detach().cpu())}
        return float(loss.detach().cpu()), metrics

    def _create_placeholder_targets(self, logits: Tensor) -> Tuple[Tensor, Tensor]:
        # This function must be replaced by your real target creation.
        # For now we create random targets to allow smoke runs.
        b, t, v = logits.shape
        targets = torch.randint(0, v, (b, t), dtype=torch.long, device=logits.device)
        mask = torch.ones((b, t), dtype=torch.float32, device=logits.device)
        return targets, mask

    def validate(self) -> float:
        if self.val_loader is None:
            return float("nan")
        self.model.eval()
        running = 0.0
        count = 0
        with torch.no_grad():
            for batch in self.val_loader:
                waveforms = batch["waveforms"].to(self.device)
                lengths = batch["lengths"].to(self.device) if "lengths" in batch else None
                out = self.model(waveforms, lengths=lengths)
                logits = out["logits"]
                targets, mask = self._create_placeholder_targets(logits)
                loss = self.compute_mlm_loss(logits, targets, mask)
                running += float(loss.detach().cpu())
                count += 1
        val_loss = running / max(1, count)
        self.writer.add_scalar("val/loss", val_loss, global_step=self.global_step)
        return val_loss

    def fit(self) -> None:
        print("Starting training loop")
        start = time.time()
        loader_iter = iter(self.train_loader)
        pbar = tqdm(total=self.cfg.total_steps)

        while self.global_step < self.cfg.total_steps:
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(self.train_loader)
                batch = next(loader_iter)

            loss, metrics = self.train_step(batch)

            # logging
            if self.global_step % self.cfg.log_every_steps == 0:
                self.writer.add_scalar("train/mlm_loss", metrics["mlm_loss"], self.global_step)
                pbar.set_postfix({"step": self.global_step, "loss": metrics["mlm_loss"]})

            # unfreeze encoder after warm_head_steps
            if self.global_step == self.cfg.warm_head_steps:
                print("Warm head stage complete. Unfreezing encoder for joint training.")
                self._unfreeze_encoder()

            # periodic checkpoint
            if self.global_step % self.cfg.save_every_steps == 0 and self.global_step > 0:
                ckpt_path = self.save_checkpoint()
                print(f"Saved checkpoint: {ckpt_path}")

            # validate periodically (simple)
            if self.global_step % (self.cfg.save_every_steps) == 0 and self.val_loader is not None:
                val_loss = self.validate()
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint(name="best.pt")

            self.global_step += 1
            pbar.update(1)

        pbar.close()
        elapsed = time.time() - start
        print(f"Training finished in {elapsed / 60:.2f} minutes. Final step: {self.global_step}")
        self.save_checkpoint(name="final.pt")


# -------------------------------
# Placeholder hook functions to be implemented by user
# -------------------------------

def load_w2vbert_encoder(checkpoint_path: Optional[str] = None, device: Optional[str] = None) -> nn.Module:
    """
    Load your w2v-BERT encoder here. This function should return a nn.Module whose forward accepts
    waveforms tensor (B, T) or (B, 1, T) and lengths and returns per-frame features (B, T', H) where T'
    aligns with the temporal resolution expected for MLM targets.

    If your checkpoint does not include an MLM head, load the encoder weights only and leave head for wrapper.
    """
    raise NotImplementedError("Please implement load_w2vbert_encoder() to load your specific encoder model/checkpoint.")


def encode_targets_with_quantizer(waveforms: Tensor, encoder: nn.Module) -> Tuple[Tensor, Tensor]:
    """
    If your MLM targets are discrete codes from a quantizer, implement this function. It should return:
      - targets: (B, T_target) LongTensor with token ids
      - mask: (B, T_target) FloatTensor mask where 1=valid, 0=ignore
    Otherwise implement your target creation logic here.
    """
    raise NotImplementedError("Please implement encode_targets_with_quantizer() for target creation.")


# -------------------------------
# CLI / entrypoint
# -------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continue pretraining w2v-BERT on new speech corpus")
    parser.add_argument("--config", type=str, required=True, help="Path to JSON config file or YAML (JSON supported here)")
    parser.add_argument("--resume", type=str, default=None, help="Optional checkpoint to resume from")
    return parser.parse_args()


def load_config(path: str) -> TrainConfig:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return TrainConfig(**data)


def build_dataloaders(cfg: TrainConfig) -> Tuple[DataLoader, Optional[DataLoader]]:
    ds = ManifestAudioDataset(cfg.manifest, sample_rate=cfg.sample_rate, min_sec=cfg.min_chunk_sec, max_sec=cfg.max_chunk_sec, shuffle=cfg.shuffle_manifest)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=collate_audio_batch, drop_last=True)
    # For simplicity we do not create a val loader here. The user can extend.
    return dl, None


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.seed)
    ensure_dir(cfg.output_dir)

    train_loader, val_loader = build_dataloaders(cfg)

    # load encoder (user-implemented loader)
    encoder = load_w2vbert_encoder(checkpoint_path=cfg.resume_from, device=cfg.device)

    model = W2VBertForContinuePretrain(encoder=encoder, hidden_size=cfg.hidden_size, vocab_size=cfg.vocab_size)

    trainer = Trainer(cfg, model, train_loader, val_loader)
    if args.resume:
        trainer.load_checkpoint(args.resume, strict=False)
    trainer.fit()


if __name__ == "__main__":
    main()
