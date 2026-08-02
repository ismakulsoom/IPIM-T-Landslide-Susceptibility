import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IPIMTDataset(Dataset):
    """
    Dataset loader for IPIM-T.

    Expected data:
    X_env_demo.npy: [samples, 15]
    X_insar_demo.npy: [samples, 8]
    y_demo.npy: [samples]
    """

    def __init__(
        self,
        env_path,
        insar_path,
        label_path
    ):
        self.env = np.load(env_path)
        self.insar = np.load(insar_path)
        self.labels = np.load(label_path)

        assert len(self.env) == len(self.insar)
        assert len(self.env) == len(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        # Load flat data
        env_flat = self.env[index]      # [15]
        insar_flat = self.insar[index]  # [8]
        label = self.labels[index]      # [1]

        # ============================================================
        # FIX: Reshape with 8x8 spatial dimensions
        # The topo_encoder and env_encoder have 3 MaxPool2d layers
        # Input needs height >= 8 and width >= 8 to survive
        # ============================================================
        
        # ============================================================
        # ENVIRONMENTAL FEATURES: [15] -> [15, 8, 8]
        # ============================================================
        env = torch.tensor(env_flat, dtype=torch.float32)
        env = env.view(15, 1, 1)        # [15, 1, 1]
        env = env.repeat(1, 8, 8)       # [15, 8, 8]
        
        # ============================================================
        # INSAR FEATURES: [8] -> [1, 8, 8]
        # First reshape to spatial grid, then expand to 8x8
        # ============================================================
        insar = torch.tensor(insar_flat, dtype=torch.float32)
        
        # Option 1: Reshape to [1, 2, 4] (1 channel, 2x4 spatial)
        # Then interpolate/expand to 8x8
        insar = insar.view(1, 2, 4)     # [1, 2, 4]
        insar = torch.nn.functional.interpolate(
            insar.unsqueeze(0),  # [1, 1, 2, 4]
            size=(8, 8),         # Target size
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # [1, 8, 8]
        
        # OR Option 2: Reshape to [1, 8, 1] and interpolate
        # insar = insar.view(1, 8, 1)
        # insar = torch.nn.functional.interpolate(
        #     insar.unsqueeze(0),
        #     size=(8, 8),
        #     mode='bilinear',
        #     align_corners=False
        # ).squeeze(0)
        
        label = torch.tensor(label, dtype=torch.float32)

        return {
            "env": env,
            "insar": insar,
            "label": label
        }


def get_dataloader(config):
    dataset = IPIMTDataset(
        config.ENV_DATA,
        config.INSAR_DATA,
        config.LABEL_DATA
    )

    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # For Windows compatibility
        pin_memory=True
    )

    return loader