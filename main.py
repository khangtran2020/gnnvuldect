from config import parse_args
from data.utils import get_data
from utils.console import console


def run(args):

    # Load Data
    console.log("Loading dataset: ", args.dataset)
    dataset = get_data(args.dataset)
    console.log("Dataset loaded, size: ", len(dataset))


if __name__ == "__main__":
    args = parse_args()
    run(args)
