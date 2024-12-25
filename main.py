from config import parse_args
from data.utils import get_data
from utils.console import console
from torch.utils.data import DataLoader


def run(args):

    # Load Data
    console.log("Loading dataset: ", args.dataset)
    dataset = get_data(args.dataset)
    console.log("Dataset loaded, size: ", len(dataset))
    console.log("Taking a look at the first element: ", dataset[0])

    # Data Loader
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    console.log("Data Loader created")
    # Get the first batch
    for batch in loader:
        console.log("Batch size: ", len(batch))
        console.log("Batch: ", batch)
        break


if __name__ == "__main__":
    args = parse_args()
    run(args)
