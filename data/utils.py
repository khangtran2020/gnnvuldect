from data.core import Data
from data.prelim_data import PrelimData


def get_data(data: str) -> Data:

    if data == "prelim":
        return PrelimData(name=data)
