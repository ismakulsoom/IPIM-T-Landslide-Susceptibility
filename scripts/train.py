
import os
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from datasets.data_loader import get_dataloader
from models.ipim_t import IPIMT
from models.physics_loss import physics_loss


def train():

    cfg=Config()

    loader=get_dataloader(cfg)

    model=IPIMT(cfg).to(cfg.DEVICE)

    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=cfg.LR
    )

    criterion=nn.BCELoss()

    os.makedirs(
        os.path.dirname(cfg.CHECKPOINT),
        exist_ok=True
    )


    best_loss=999


    for epoch in range(cfg.EPOCHS):

        model.train()

        total=0


        for batch in tqdm(loader):

            env=batch["env"].to(cfg.DEVICE)

            insar=batch["insar"].to(cfg.DEVICE)

            label=batch["label"].to(cfg.DEVICE)


            optimizer.zero_grad()


            pred=model(
                env,
                insar
            )


            loss=criterion(
                pred,
                label
            )


            loss.backward()

            optimizer.step()


            total+=loss.item()


        avg=total/len(loader)

        print(
            f"Epoch {epoch+1}: loss={avg:.4f}"
        )


        if avg < best_loss:

            best_loss=avg

            torch.save(
                model.state_dict(),
                cfg.CHECKPOINT
            )


if __name__=="__main__":
    train()
