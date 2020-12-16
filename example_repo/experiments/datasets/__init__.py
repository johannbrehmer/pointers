from .images import get_fashion_mnist_dataset, get_imagenet_dataset
from .uci import get_uci_dataset


def get_dataset(dataset_name, directory, partition="train", valid_fraction=0.05):
    if dataset_name == "fashion-mnist":
        return get_fashion_mnist_dataset(partition, directory, valid_fraction)
    elif dataset_name == "imagenet":
        return get_imagenet_dataset(partition, directory, valid_fraction)
    elif dataset_name in ["miniboone", "gas", "hepmass", "bsds300", "power"]:
        return get_uci_dataset(dataset_name, partition, directory)
    else:
        raise RuntimeError(f"Unknown dataset {dataset_name}")


def get_dataset_resolution(dataset_name):
    RESOLUTIONS = {
        "fashion-mnist": (1, 32, 32),
        "imagenet": (3, 64, 64),
        "miniboone": (43,),
        "gas": (8,),
        "hepmass": (21,),
        "bsds300": (63,),
        "power": (6,),
    }
    return RESOLUTIONS[dataset_name]
