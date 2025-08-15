from dataclasses import dataclass, asdict

@dataclass
class Config:
    gpus: str = "0"
    
    seed: int = 42
    data_root: str = "SIDD_Small_sRGB_Only"
    scene_list_file: str = "SIDD_Small_sRGB_Only/Scene_Instances.txt"
    data_path: str = "SIDD_Small_sRGB_Only/Data/"

    filter_val: int = 128
    
    num_noisy: int = 6
    crop_size: int = 256

    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True

    lr: float = 1e-3
    step_size: int = 10
    gamma: float = 0.1
    num_epochs: int = 100

    checkpoint_every: int = 0           
    best_metric: str = "val_psnr"        
    log_dir: str = "logs"

    model_name: str = "UNet"
    run_suffix: str = ""   

    enable_profiling: bool = True
    
cfg = Config()
