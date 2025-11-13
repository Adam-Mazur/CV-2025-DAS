import argparse


if __name__ == "__main__":
    methods = {
        "hough": ("hough_method", "HoughMethod"),
        "hough_prob": ("hough_prob_method", "HoughProbMethod"),
    }
    transforms = {
        "abs": "AbsoluteValue",
        "normalize": "Normalize",
        "clip": "Clip",
        "zscore": "ZScoreTransform",
        "median": "MedianFilter",
        "tv": "TotalVariationDenoising",
        "resize": "Resize",
        "nlm": "NonLocalMeansDenoising",
    }

    parser = argparse.ArgumentParser(
        description="Detect vehicle velocity from DAS data."
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=methods.keys(),
        help="Method to use for detection.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="*",
        choices=transforms.keys(),
        help="Data transforms to apply before detection.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to save the output visualization.",
    )
    parser.add_argument(
        "--start-time",
        type=str,
        required=True,
        help="Start time for analysis (HH:MM:SS).",
    )
    parser.add_argument(
        "--end-time", type=str, required=True, help="End time for analysis (HH:MM:SS)."
    )
    parser.add_argument(
        "--group-lines",
        action="store_true",
        help="Whether to group similar detected lines.",
    )
    args = parser.parse_args()

    print("Importing modules...")
    from src.visualize import visualize_lines
    from src.group_lines import group_lines, group_segments
    from src.method import Method
    from src.get_data import get_data
    from datetime import time
    import src.transforms
    import importlib
    import yaml

    with open("src/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    start_time = time.fromisoformat(args.start_time)
    end_time = time.fromisoformat(args.end_time)

    print("Loading data...")
    df = get_data(start_time, end_time, ignore_missing=True)

    print("Applying transforms...")
    transform_objects = []
    if args.transforms:
        for t_name in args.transforms:
            t_class = getattr(src.transforms, transforms[t_name])
            t_config = config["transforms"].get(t_name, {})
            t_obj = t_class(**t_config)
            transform_objects.append(t_obj)

    data = df.copy()
    for t in transform_objects:
        data = t.apply(data)

    method_module_name, method_class_name = methods[args.method]
    method_cls = getattr(
        importlib.import_module("src." + method_module_name), method_class_name
    )
    method_config = config["methods"].get(args.method, {})
    method_obj: Method = method_cls(**method_config)

    print("Detecting lines...")
    lines = method_obj.detect(data)

    if args.group_lines:
        print("Grouping lines...")
        if len(lines[0]) == 4:
            lines = group_segments(
                lines, data.shape[1], **config.get("group_segments", {})
            )
        else:
            lines = group_lines(lines, data.shape[1], **config.get("group_lines", {}))

    print("Visualizing results...")
    visualize_lines(data, lines, args.output)
