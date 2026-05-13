from typing import Dict, Tuple, List
from pathlib import Path


def parse_learning_curve_file(learning_curve_fpath: str) -> Dict:
    """
    Parse n2p2 learning-curve.out file.

    Args:
        learning_curve_fpath (str): Path to the learning-curve.out file.

    Returns:
        Dict: Parsed learning curve data as a dictionary.
    """
    content = Path(learning_curve_fpath).read_text()
    return parse_learning_curve_content(content)


def parse_learning_curve_content(learning_curve_content: str) -> Dict:
    """
    Parse n2p2 learning-curve.out content.

    Args:
        learning_curve_content (str): Content of the learning-curve.out file.

    Returns:
        Dict: Parsed learning curve data as a dictionary.
    """
    columns = [
        'epoch',
        "RMSEpa_Etrain_pu",
        "RMSEpa_Etest_pu",
        "RMSE_Etrain_pu",
        "RMSE_Etest_pu",
        "MAEpa_Etrain_pu",
        "MAEpa_Etest_pu",
        "MAE_Etrain_pu",
        "MAE_Etest_pu",
    ]
    if "MAE_Ftest_pu" in learning_curve_content:
        columns += [
            "RMSE_Ftrain_pu",
            "RMSE_Ftest_pu",
            "MAE_Ftrain_pu",
            "MAE_Ftest_pu",
        ]

    lines = learning_curve_content.split('\n')
    epochs_data = {}

    for line in lines:
        if not line or line.startswith("#"):
            continue

        values = [float(v) for v in line.split()]
        epoch = int(values[0])
        items = zip(columns[1:], values[1:])
        epochs_data[epoch] = dict(items)

    return epochs_data


def select_best_epoch_from_dict(
    epoch_data: Dict,
    quantity: str = "energy",
    metric: str = "MAE",
    dataset: str = "test",
    min_epoch: int = 1,
    per_atom: bool = False,
    **kwargs
) -> Tuple[int, List[float]]:
    """
    Select best epoch from n2p2 learning-curve.out content.

    Args:
        epoch_data (Dict): Parsed learning curve data as a dictionary.
        quantity (str): The quantity to select the best epoch for.
        metric (str): The error metric to use for judging the best epoch.
        dataset (str): The dataset to consider.
        min_epoch (int): The minimum epoch to consider for selection.
        per_atom (bool): If True, use energies per atom rather than total.

    Returns:
        Tuple[int, List[float]]: The best epoch and a list of values for the specified quantity.
    """
    metric = metric.upper()
    pa = "pa" if per_atom else ''

    methods = {
        'energy': lambda x: epoch_data[x][f"{metric}{pa}_E{dataset}_pu"],
        'forces': lambda x: epoch_data[x][f"{metric}_F{dataset}_pu"],
        'combined': (
            lambda x:
            epoch_data[x][f"{metric}{pa}_E{dataset}_pu"] +
            epoch_data[x][f"{metric}_F{dataset}_pu"]
        ),
    }

    fn = methods[quantity]

    try:
        best_epoch = min(
            {x: epoch_data[x] for x in epoch_data if x >= min_epoch},
            key=fn,
        )
    except KeyError as e:
        raise ValueError(f"the data does not contain the quantity specified: {e}")

    return best_epoch, list(map(fn, epoch_data))


def select_best_epoch_from_file(
    learning_curve_fpath: str = "learning-curve.out", **kwargs
) -> Tuple[int, List[float]]:
    """
    Select best epoch from n2p2 learning-curve.out file.

    Args:
        learning_curve_fpath (str): Path to the learning-curve.out file.
        **kwargs: Additional keyword arguments to pass to the selection function.

    Returns:
        Tuple[int, List[float]]: The best epoch and a list of values for the specified quantity.
    """
    epoch_data = parse_learning_curve_file(learning_curve_fpath)
    return select_best_epoch_from_dict(epoch_data, **kwargs)


def select_best_epoch_from_string(
    learning_curve_content: str, **kwargs
) -> Tuple[int, List[float]]:
    """
    Select best epoch from n2p2 learning-curve.out content string.

    Args:
        learning_curve_content (str): The content of the learning-curve.out file.
        **kwargs: Additional keyword arguments to pass to the selection function.

    Returns:
        Tuple[int, List[float]]: The best epoch and a list of values for the specified quantity.
    """
    epoch_data = parse_learning_curve_content(learning_curve_content)
    return select_best_epoch_from_dict(epoch_data, **kwargs)

    return select_best_epoch_from_dict(epoch_data, **kwargs)


def main():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "filepath",
        type=str,
        help="The learning-curve.out filepath",
    )

    parser.add_argument(
        "-q",
        "--quantity",
        type=str,
        choices=["energy", "forces", "combined"],
        default="energy",
        help="Quantity used to select best epoch",
    )

    parser.add_argument(
        "-e",
        "--error-metric",
        choices=["rmse", "mae"],
        default="mae",
        help="Error metric for judging best epoch",
    )

    parser.add_argument(
        "-pa",
        "--per-atom",
        action="store_true",
        default=False,
        help="Use energies/forces per atom rather than total",
    )

    parser.add_argument(
        "-me",
        "--min-epoch",
        type=int,
        default=1,
        help="Minimum epoch able to select as best",
    )

    args = parser.parse_args()

    best_epoch, values = select_best_epoch_from_file(
        learning_curve_fpath=args.filepath,
        quantity=args.quantity,
        metric=args.error_metric,
        per_atom=args.per_atom,
        min_epoch=args.min_epoch,
    )

    print(
        f"Best epoch for {args.quantity} "
        f"using metric {args.error_metric.upper()}{'pa' if args.per_atom else ''} "
        f"is epoch {best_epoch} "
        f"({values[best_epoch]})"
    )

if __name__ == "__main__":
    main()
