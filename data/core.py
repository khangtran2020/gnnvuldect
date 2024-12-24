from torch.utils.data import Dataset, DataLoader


class Data(Dataset):

    def __init__(self, name: str) -> None:
        super().__init__(name=name)

    def process(self):
        pass

    def __getitem__(self, idx):
        pass

    def __len__(self):
        pass
