from config import parse_args
from data.utils import get_data
from utils.console import console


def run(args):

    # Load Data
    console.log("Loading dataset: ", args.dataset)
    dataset = get_data(args.dataset)
    # dataset.
    console.log("Dataset loaded, size: ", len(dataset))
    console.log("Taking a look at the first element: ", dataset[0])


if __name__ == "__main__":
    args = parse_args()
    run(args)
