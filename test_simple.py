"""
Simple test for IPIM-T that bypasses DataLoader.
"""
import sys
import torch
import numpy as np
import math

sys.path.append(".")

from config import Config
from models.ipim_t import IPIMT

def main():
    print("=" * 60)
    print("IPIM-T Simple Test (Bypasses DataLoader)")
    print("=" * 60)
    
    # Step 1: Load configuration
    try:
        cfg = Config()
        print("✅ Config loaded successfully!")
        print(f"   BATCH_SIZE: {cfg.BATCH_SIZE}")
        print(f"   INPUT_ENV_DIM: {cfg.INPUT_ENV_DIM}")
        print(f"   INPUT_INSAR_DIM: {cfg.INPUT_INSAR_DIM}")
    except Exception as e:
        print(f"❌ Config error: {e}")
        return
    
    # Step 2: Create test data
    batch_size = 4
    
    # ============================================================
    # ENVIRONMENTAL FEATURES: 15 features
    # Reshape to [batch, 15, 16, 16] for Conv2d
    # The encoder has 3 MaxPool2d layers, so we need H>=8 and W>=8
    # ============================================================
    spatial_size = 16  # Use 16x16 to be safe
    env_flat = torch.randn(batch_size, 15)
    env_features = env_flat.unsqueeze(-1).unsqueeze(-1)  # [batch, 15, 1, 1]
    env_features = env_features.repeat(1, 1, spatial_size, spatial_size)  # [batch, 15, 16, 16]
    
    # ============================================================
    # INSAR FEATURES: 8 features
    # Reshape to [batch, 1, 16, 16] for InSARTransformerEncoder
    # ============================================================
    insar_flat = torch.randn(batch_size, 8)
    insar_features = insar_flat.view(batch_size, 1, 1, 1)
    insar_features = insar_features.repeat(1, 1, spatial_size, spatial_size)  # [batch, 1, 16, 16]
    
    print(f"\n✅ Created test data:")
    print(f"   env_features shape: {env_features.shape}")
    print(f"   insar_features shape: {insar_features.shape}")
    print(f"   Using {spatial_size}x{spatial_size} spatial grid")
    
    # Step 3: Initialize model
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = IPIMT(cfg)
        model = model.to(device)
        print(f"\n✅ Model initialized successfully!")
        print(f"   Device: {device}")
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return
    
    # Step 4: Forward pass
    try:
        env_features = env_features.to(device)
        insar_features = insar_features.to(device)
        
        print(f"\n   Before forward - env shape: {env_features.shape}")
        print(f"   Before forward - insar shape: {insar_features.shape}")
        
        output = model(env_features, insar_features)
        
        print(f"\n✅ Forward pass successful!")
        print(f"   Output shape: {output.shape}")
    except Exception as e:
        print(f"❌ Forward pass error: {e}")
        return
    
    print("\n" + "=" * 60)
    print("✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()