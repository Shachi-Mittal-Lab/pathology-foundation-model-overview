import torch
import yaml
from pathlib import Path
from torch.utils.data import DataLoader
from networks.dpt.models import PathOrchestraDPT
from data.dataset import HEdataset

def find_max_batch_size(config_path, start_batch_size=32, max_attempts=10):
    """
    Binary search to find maximum batch size that fits in GPU memory.
    
    INPUTS:
    - config_path (str): path to config.yaml
    - start_batch_size (int): initial batch size to try
    - max_attempts (int): max iterations before giving up
    
    OUTPUTS:
    - max_batch_size (int): largest batch size that fits
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
    
    # Load model
    model = PathOrchestraDPT(
        hf_token=cfg["hf_token"],
        num_classes=cfg["num_classes"],
        features=cfg.get("features", 256),
        freeze_encoder=cfg.get("freeze_encoder", True),
    ).to(device)
    model.eval()
    
    # Load dataset
    all_files = sorted(Path(cfg["data_root"]).glob("*.npz"))
    if len(all_files) < 10:
        print(f"Warning: Only {len(all_files)} .npz files found. Using all for testing.")
        test_files = all_files
    else:
        test_files = all_files[:10]  # Use first 10 for speed
    
    dataset = HEdataset(cfg["data_root"], split="train", file_list=test_files)
    
    batch_size = start_batch_size
    max_batch_size = 0
    attempt = 0
    
    while attempt < max_attempts:
        try:
            print(f"Attempt {attempt + 1}: Testing batch_size={batch_size}...", end=" ")
            
            # Clear cache
            if device.type == "cuda":
                torch.cuda.empty_cache()
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,  # Disable workers for testing
                pin_memory=True
            )
            
            # Try one forward pass
            with torch.no_grad():
                for imgs, masks in dataloader:
                    imgs = imgs.to(device)
                    _ = model(imgs)
                    break
            
            print(f"✓ SUCCESS")
            max_batch_size = batch_size
            batch_size *= 2  # Try larger batch size
            attempt += 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ OOM")
                # Reduce batch size
                if batch_size == 1:
                    print("Cannot even fit batch_size=1!")
                    break
                batch_size = max(1, batch_size // 2)
                attempt += 1
            else:
                print(f"✗ Error: {e}")
                break
    
    print(f"\n{'='*60}")
    print(f"Maximum batch size: {max_batch_size}")
    print(f"{'='*60}\n")
    
    # Show memory usage at max batch size
    try:
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
        
        dataloader = DataLoader(
            dataset,
            batch_size=max_batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )
        
        with torch.no_grad():
            for imgs, masks in dataloader:
                imgs = imgs.to(device)
                _ = model(imgs)
                break
        
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated(0) / 1e9
            print(f"Peak GPU memory at batch_size={max_batch_size}: {peak_memory:.2f} GB")
    except:
        pass
    
    return max_batch_size


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--start_batch_size", type=int, default=32)
    args = parser.parse_args()
    
    max_bs = find_max_batch_size(args.config, start_batch_size=args.start_batch_size)
    print(f"Recommendation: Use batch_size={max_bs} or slightly lower (e.g., {max(1, max_bs-4)})")
