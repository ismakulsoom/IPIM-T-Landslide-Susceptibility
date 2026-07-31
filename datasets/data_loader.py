
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class IPIMTDataset(Dataset):
    """
    Dataset loader for IPIM-T.

    Expected data:
    X_env_demo.npy
    X_insar_demo.npy
    y_demo.npy

    Environmental and InSAR streams are kept separate
    because IPIM-T uses multi-stream feature extraction.
    """

    def __init__(
        self,
        env_path,
        insar_path,
        label_path
    ):

        self.env=np.load(env_path)
        self.insar=np.load(insar_path)
        self.labels=np.load(label_path)


        assert len(self.env)==len(self.insar)
        assert len(self.env)==len(self.labels)


    def __len__(self):
        return len(self.labels)


    def __getitem__(self,index):

        env=torch.tensor(
            self.env[index],
            dtype=torch.float32
        )

        insar=torch.tensor(
            self.insar[index],
            dtype=torch.float32
        )

        label=torch.tensor(
            self.labels[index],
            dtype=torch.float32
        )


        return {
            "env":env,
            "insar":insar,
            "label":label
        }



def get_dataloader(config):

    dataset=IPIMTDataset(
        config.ENV_DATA,
        config.INSAR_DATA,
        config.LABEL_DATA
    )


    loader=DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


    return loader
