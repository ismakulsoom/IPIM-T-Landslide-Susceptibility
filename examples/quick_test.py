
from config import Config
from datasets.data_loader import get_dataloader
from models.ipim_t import IPIMT


def main():

    cfg=Config()

    loader=get_dataloader(cfg)

    batch=next(iter(loader))


    model=IPIMT(cfg)

    prediction=model(
        batch["env"],
        batch["insar"]
    )


    print("======================")
    print("IPIM-T quick test")
    print("======================")

    print(
        "Prediction shape:",
        prediction.shape
    )

    print(
        "Test completed successfully"
    )


if __name__=="__main__":
    main()
