
import torch


def factor_of_safety(
    slope,
    cohesion,
    friction,
    gamma,
    depth
):

    slope=torch.deg2rad(slope)

    phi=torch.deg2rad(friction)

    numerator = (
        cohesion +
        gamma*depth*
        torch.cos(slope)**2*
        torch.tan(phi)
    )

    denominator = (
        gamma*depth*
        torch.sin(slope)*
        torch.cos(slope)
        +1e-6
    )

    return numerator/denominator



def physics_loss(
    probability,
    fs,
    threshold=0.5,
    weight=0.1
):

    mask=(
        (fs>1.5)
        &
        (probability>threshold)
    ).float()


    return weight*torch.mean(
        mask*
        (probability-threshold)**2
    )
