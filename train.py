# ---- САМЫЙ ВЕРХ ФАЙЛА, сразу после импорта cfg ----
from config import cfg, Config
import os, logging
from pathlib import Path
import time, json
from dataclasses import dataclass, asdict

if getattr(cfg, "gpus", ""):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpus

def setup_run_folders(cfg: Config):
    ts = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"{cfg.model_name}_{ts}"
    if getattr(cfg, "run_suffix", ""):
        run_name += f"_{cfg.run_suffix}"
    base = Path(cfg.log_dir) / run_name
    (base / "ckpts").mkdir(parents=True, exist_ok=True)
    (base / "plots").mkdir(parents=True, exist_ok=True)
    (base / "preds").mkdir(parents=True, exist_ok=True)
    return base

run_dir = setup_run_folders(cfg)
ckpt_dir = run_dir / "ckpts"
plots_dir = run_dir / "plots"
preds_dir = run_dir / "preds"

def setup_logging(run_dir: Path):
    log_file = run_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler()
        ],
        force=True,         
    )
    logging.captureWarnings(True)  
    logging.info("Логирование инициализировано.")
    logging.info(f"Log file: {log_file.resolve()}")
    return log_file

log_file = setup_logging(run_dir)

with open(run_dir / "config.json", "w", encoding="utf-8") as f:
    json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

import random, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from datamodule import MultiImagePatchesDataset
from model import HalfUNet

from torchmetrics.functional import structural_similarity_index_measure as tm_ssim
import pickle, gzip, hashlib

    
USE_DATASET_CACHE = True
CACHE_VERSION = "v1"  

with open(cfg.scene_list_file) as f:
    instances = f.read().splitlines()

noisy_images_path, gt_images_path = [], []
for fscene in instances:
    p = os.path.join(cfg.data_path, fscene)
    for g in os.listdir(p):
        image_path = os.path.join(p, g)
        if "NOISY" in image_path:
            noisy_images_path.append(image_path)
        else:
            gt_images_path.append(image_path)

logging.info(f"TOTAL NOISY IMAGES: {len(noisy_images_path)}")
logging.info(f"TOTAL GROUND TRUTH IMAGES: {len(gt_images_path)}")

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _dataset_cache_key(noisy_list, gt_list, cfg):
    h = hashlib.sha1()
    for p in sorted(noisy_list):
        h.update(p.encode("utf-8"))
    h.update(b"|")
    for p in sorted(gt_list):
        h.update(p.encode("utf-8"))
    h.update(f"|num_noisy={cfg.num_noisy}|crop={cfg.crop_size}|{CACHE_VERSION}".encode("utf-8"))
    return h.hexdigest()[:16]

cache_dir = Path(cfg.data_root) / "cache"
cache_dir.mkdir(parents=True, exist_ok=True)
cache_key = _dataset_cache_key(noisy_images_path, gt_images_path, cfg)
cache_file = cache_dir / f"dataset_{cache_key}.pkl.gz"

dataset = None

if USE_DATASET_CACHE and cache_file.exists():
    try:
        with gzip.open(cache_file, "rb") as f:
            dataset = pickle.load(f)
        logging.info(f"Dataset загружен из кэша: {cache_file.name}")
    except Exception as e:
        logging.warning(f"Не смог загрузить кэш датасета ({e}). Пересобираю...")

if dataset is None:
    dataset = MultiImagePatchesDataset(
        noisy_images=noisy_images_path,
        gt_images=gt_images_path,
        num_noisy=cfg.num_noisy,
        crop_size=cfg.crop_size
    )
    if USE_DATASET_CACHE:
        try:
            with gzip.open(cache_file, "wb") as f:
                pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
            logging.info(f"Dataset сохранён в кэш: {cache_file.name}")
        except Exception as e:
            logging.warning(f"Не удалось сохранить датасет в кэш: {e}")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_into(model, ckpt_path, map_location="cpu", strict=True):
    state = torch.load(ckpt_path, map_location=map_location)
    sd = state["state_dict"]
    has_module = next(iter(sd)).startswith("module.")

    if is_dp and not has_module:
        sd = {f"module.{k}": v for k, v in sd.items()}
        model.load_state_dict(sd, strict=strict)
    elif (not is_dp) and has_module:
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
        model.load_state_dict(sd, strict=strict)
    else:
        model.load_state_dict(sd, strict=strict)
    return state

train_size = int(0.8 * len(dataset))
valid_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - valid_size
train_dataset, valid_dataset, test_dataset = random_split(
    dataset, [train_size, valid_size, test_size],
    generator=torch.Generator().manual_seed(cfg.seed)
)

train_loader = DataLoader(
    train_dataset, batch_size=cfg.batch_size, shuffle=True,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
)
valid_loader = DataLoader(
    valid_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
)
test_loader = DataLoader(
    test_dataset, batch_size=cfg.batch_size, shuffle=False,
    num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
)

for noisy_in, gt_out in train_loader:
    logging.info(f"Noisy input shape: {tuple(noisy_in.shape)}")  
    logging.info(f"GT shape: {tuple(gt_out.shape)}")             
    break

logging.info(f"Size of training set: {len(train_dataset)}")
logging.info(f"Size of validation set: {len(valid_dataset)}")
logging.info(f"Size of test set: {len(test_dataset)}")

model = HalfUNet(input_channels=3 * cfg.num_noisy)

model = model.to(device)
model.eval()

logging.info(f"MODEL:\n{model}")
print(model)

def get_model_params(model: nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

total_params, trainable_params = get_model_params(model)
logging.info(f"Total parameters: {total_params:,}")
logging.info(f"Trainable parameters: {trainable_params:,}")
with open(run_dir / "model_params.txt", "w", encoding="utf-8") as f:
    f.write(f"Total parameters: {total_params}\nTrainable parameters: {trainable_params}\n")

def psnr(img1, img2, data_range=1.0):
    mse = F.mse_loss(img1, img2)
    return 20.0 * torch.log10(data_range / torch.sqrt(mse + 1e-12))

def loss_func(pred, target):
    return 100.0 - psnr(pred, target)

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = total_psnr = total_ssim = 0.0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss_val = loss_func(outputs, targets)
        psnr_val = psnr(outputs, targets)
        ssim_val = tm_ssim(outputs, targets, data_range=1.0)
        total_loss += loss_val.item()
        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()
    n = len(dataloader)
    return total_loss / n, total_psnr / n, total_ssim / n

optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
                

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_size, gamma=cfg.gamma)

def unwrap_state_dict(m: nn.Module):
    return m.state_dict()

def save_checkpoint(epoch, model, optimizer, scheduler, best_metric_value, tag):
    state = {
        "epoch": epoch,
        "state_dict": unwrap_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "best_metric_value": best_metric_value,
        "config": asdict(cfg),
    }
    path = ckpt_dir / f"{tag}.pth"
    torch.save(state, path)
    logging.info(f"Сохранён чекпойнт: {path.name}")

def plot_and_save_curves(epochs, train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim):
    plt.figure(figsize=(6,6))
    plt.plot(epochs, train_loss, label='Train Loss')
    plt.plot(epochs, val_loss, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(epochs, train_psnr, label='Train PSNR')
    plt.plot(epochs, val_psnr, label='Val PSNR')
    plt.title('PSNR')
    plt.xlabel('Epoch'); plt.ylabel('PSNR (dB)'); plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "psnr.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6,6))
    plt.plot(epochs, train_ssim, label='Train SSIM')
    plt.plot(epochs, val_ssim, label='Val SSIM')
    plt.title('SSIM')
    plt.xlabel('Epoch'); plt.ylabel('SSIM'); plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "ssim.png", dpi=150)
    plt.close()

metrics_csv_path = run_dir / "metrics.csv"
with open(metrics_csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "epoch", "lr",
        "train_loss", "train_psnr", "train_ssim",
        "val_loss", "val_psnr", "val_ssim",
        "epoch_time_s", "elapsed_min"
    ])

best_val_metric = -float("inf") if cfg.best_metric in ("val_psnr", "val_ssim") else float("inf")
history = {
    "train_loss": [], "train_psnr": [], "train_ssim": [],
    "val_loss": [], "val_psnr": [], "val_ssim": []
}

cuda_sync()
train_wall_start = time.perf_counter()

for epoch in range(1, cfg.num_epochs + 1):
    cuda_sync()
    epoch_start = time.perf_counter()
    
    model.train()
    total_loss = total_psnr = total_ssim = 0.0

    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch}/{cfg.num_epochs}")
    for batch_idx, (inputs, targets) in pbar:
        inputs, targets = inputs.to(device), targets.to(device)

        outputs = model(inputs)
        loss = loss_func(outputs, targets)
        psnr_val = psnr(outputs, targets)
        ssim_val = tm_ssim(outputs, targets, data_range=1.0)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_psnr += psnr_val.item()
        total_ssim += ssim_val.item()

        pbar.set_postfix(loss=f"{loss.item():.3f}", psnr=f"{psnr_val.item():.2f}", ssim=f"{ssim_val.item():.3f}")

    n_train = len(train_loader)
    avg_loss = total_loss / n_train
    avg_psnr = total_psnr / n_train
    avg_ssim = total_ssim / n_train

    val_loss, val_psnr, val_ssim = evaluate(model, valid_loader, device)

    current_lr = optimizer.param_groups[0]["lr"]
    logging.info(
        f"Epoch {epoch:03d} | lr={current_lr:.6f} | "
        f"train: loss={avg_loss:.4f}, psnr={avg_psnr:.2f}, ssim={avg_ssim:.4f} | "
        f"val: loss={val_loss:.4f}, psnr={val_psnr:.2f}, ssim={val_ssim:.4f}"
    )

    history["train_loss"].append(avg_loss)
    history["train_psnr"].append(avg_psnr)
    history["train_ssim"].append(avg_ssim)
    history["val_loss"].append(val_loss)
    history["val_psnr"].append(val_psnr)
    history["val_ssim"].append(val_ssim)

    scheduler.step()

    if cfg.checkpoint_every > 0 and (epoch % cfg.checkpoint_every == 0):
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_metric, tag=f"epoch_{epoch:03d}")

    current_metric = {"val_psnr": val_psnr, "val_ssim": val_ssim, "val_loss": -val_loss}[cfg.best_metric]
    if current_metric > best_val_metric:
        best_val_metric = current_metric
        save_checkpoint(epoch, model, optimizer, scheduler, best_val_metric, tag="best")

    epochs_range = list(range(1, len(history["train_loss"]) + 1))
    plot_and_save_curves(
        epochs_range,
        history["train_loss"], history["val_loss"],
        history["train_psnr"], history["val_psnr"],
        history["train_ssim"], history["val_ssim"]
    )
    
    cuda_sync()
    epoch_time = time.perf_counter() - epoch_start
    elapsed_total = time.perf_counter() - train_wall_start
    logging.info(f"⏱ Epoch {epoch:03d} duration: {epoch_time:.2f}s | total elapsed: {elapsed_total/60:.2f} min")
    
    with open(metrics_csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            epoch, current_lr,
            avg_loss, avg_psnr, avg_ssim,
            val_loss, val_psnr, val_ssim,
            epoch_time, elapsed_total/60.0
        ])

cuda_sync()
total_time = time.perf_counter() - train_wall_start
avg_epoch_time = total_time / max(1, len(history["train_loss"]))
logging.info(f"🏁 Total training time: {total_time/60:.2f} min ({total_time:.2f} s) | avg/epoch: {avg_epoch_time:.2f} s")
print(f"Total training: {total_time/60:.2f} min, avg/epoch: {avg_epoch_time:.2f} s")

torch.cuda.empty_cache()

best_path = ckpt_dir / "best.pth"

if best_path.exists():
    state = load_checkpoint_into(model, best_path, map_location="cpu")
    logging.info(f"Загружен лучший чекпойнт: {best_path.name}")
    
model = model.to("cpu").eval()

def to_np_img(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().float().clamp(0.0, 1.0).permute(1, 2, 0).numpy()

import math
for batch_idx, (lr_batch, hr_batch) in enumerate(test_loader):
    with torch.no_grad():
        gen_hr = model(lr_batch)

    num_samples = min(5, len(lr_batch))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    if num_samples == 1:
        axes = axes[None, :]

    for i in range(num_samples):
        lr_img  = to_np_img(lr_batch[i, 6:9, :, :])
        hr_img  = to_np_img(hr_batch[i])
        gen_img = to_np_img(gen_hr[i])

        axes[i, 0].imshow(lr_img);  axes[i, 0].set_title("Noisy");         axes[i, 0].axis("off")
        axes[i, 1].imshow(hr_img);  axes[i, 1].set_title("Ground Truth");  axes[i, 1].axis("off")
        axes[i, 2].imshow(gen_img); axes[i, 2].set_title("Generated");     axes[i, 2].axis("off")

    plt.tight_layout()
    out_path = preds_dir / f"test_batch_{batch_idx:03d}.png"
    plt.savefig(out_path, dpi=150)
    plt.close()
    logging.info(f"Сохранён пример предсказаний: {out_path.name}")

    if batch_idx >= 1: 
        break

test_loss, test_psnr, test_ssim = evaluate(model, test_loader, device=torch.device("cpu"))
logging.info(f"TEST | loss={test_loss:.4f}, psnr={test_psnr:.2f}, ssim={test_ssim:.4f}")
with open(run_dir / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump({"loss": test_loss, "psnr": test_psnr, "ssim": test_ssim}, f, indent=2, ensure_ascii=False)

print(f"🗂️ Логи и артефакты: {run_dir}")
print(f"📈 Графики:         {plots_dir}")
print(f"💾 Чекпойнты:       {ckpt_dir}")
print(f"🖼️ Предсказания:    {preds_dir}")
