import yaml
import torch


class Config:

    def __init__(
        self,
        path="configs/config.yaml"
    ):

        with open(path, "r") as f:
            cfg = yaml.safe_load(f)


        # Data paths
        self.ENV_DATA = cfg["data"]["env"]

        self.INSAR_DATA = cfg["data"]["insar"]

        self.LABEL_DATA = cfg["data"]["label"]


        # Model
        self.TRANSFORMER_DIM = (
            cfg["model"]["transformer_dim"]
        )

        self.TRANSFORMER_HEADS = (
            cfg["model"]["transformer_heads"]
        )

        self.TRANSFORMER_LAYERS = (
            cfg["model"]["transformer_layers"]
        )

        self.DROPOUT = (
            cfg["model"]["dropout"]
        )


        # Training
        self.BATCH_SIZE = (
            cfg["training"]["batch_size"]
        )

        self.EPOCHS = (
            cfg["training"]["epochs"]
        )

        self.LR = (
            cfg["training"]["learning_rate"]
        )


        # Physics
        self.USE_PHYSICS = (
            cfg["physics"]["enabled"]
        )

        self.PHYSICS_LAMBDA = (
            cfg["physics"]["lambda"]
        )


        # Device
        self.DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        # Feature index
        self.TOPO_INDICES = [
            0,1,2,3,4,5,6,7
        ]

        self.ENV_INDICES = [
            8,9,10,11,12,13
        ]
