
import numpy as np
import torch

from config import Config
from datasets.data_loader import get_dataloader
from models.ipim_t import IPIMT
from utils.metrics import compute_metrics


def evaluate():

    cfg=Config()

    loader=get_dataloader(cfg)


    model=IPIMT(cfg).to(cfg.DEVICE)

    model.load_state_dict(
        torch.load(
            cfg.CHECKPOINT,
            map_location=cfg.DEVICE
        )
    )

    model.eval()


    probs=[]
    labels=[]


    with torch.no_grad():

        for batch in loader:

            p=model(
                batch["env"].to(cfg.DEVICE),
                batch["insar"].to(cfg.DEVICE)
            )

            probs.extend(
                p.cpu().numpy()
            )

            labels.extend(
                batch["label"].numpy()
            )


    print(
        compute_metrics(
            np.array(probs),
            np.array(labels)
        )
    )


if __name__=="__main__":
    evaluate()
