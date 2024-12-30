from torch import Tensor
from data.core import Data
from data.prelim_data import PrelimData


def get_data(data: str, mode: str) -> Data:

    if data == "prelim":
        return PrelimData(name=data, mode=mode)


def custom_collate(original_batch):
    filtered_data = []
    filtered_target = []

    for item in original_batch:

        filtered_data.append(item[:-1])
        filtered_target.append(item[-1])

    return filtered_data, filtered_target
