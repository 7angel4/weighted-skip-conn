import argparse
from utils.config_loader import load_config
from models.test import train_and_test
from models.train import TRAIN_LAYERS

def parse_list(value):
    """
    Parse a comma-separated string into a list of integers.
    Example: "1,2,3" -> [1, 2, 3]
    """
    try:
        return [int(item) for item in value.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError("test_layers must be a comma-separated list of integers.")

def main():
    parser = argparse.ArgumentParser(description="Weighted Skip Connections: Experiment CLI")
    
    # CLI Arguments for model_config
    parser.add_argument("--config", type=str, help="Path to the YAML configuration file")
    parser.add_argument("--model_name", type=str, help="Model name (overrides config)")
    parser.add_argument("--init_res_weight", type=float, help="Initial residual weight (overrides config)")
    
    # CLI Arguments for train_config
    parser.add_argument("--dataset", type=str, help="Dataset name (overrides config)")
    parser.add_argument("--lr", type=float, help="Learning rate (overrides config)")
    parser.add_argument("--weight_decay", type=float, help="Weight decay (overrides config)")
    parser.add_argument("--epochs", type=int, help="Number of training epochs (overrides config)")
    parser.add_argument("--max_patience", type=int, help="Maximum patience for early stopping (overrides config)")
    
    # CLI Arguments for experiment_config
    parser.add_argument("--layers", type=str, help="Comma-separated list of integers, \
                        representing number of layers to train and test for the model (overrides config)")
    parser.add_argument("--report_per_period", type=int, help="Report training status every N epochs (overrides config)")
    parser.add_argument("--export", action="store_true", help="Export results (overrides config)")

    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = load_config(args.config)
    else:
        raise ValueError("Please provide a configuration file using the --config argument.")
    
    # Separate model and train configs
    model_config = config.get("model_config", {})
    train_config = config.get("train_config", {})
    experiment_config = config.get("experiment_config", {})
    
    # Override model_config parameters from CLI if provided
    if args.model_name:
        model_config["model_name"] = args.model_name
    if args.init_res_weight is not None:
        model_config["init_res_weight"] = args.init_res_weight
    
    # Override train_config parameters from CLI if provided
    if args.dataset:
        train_config["dataset"] = args.dataset
    if args.lr:
        train_config["lr"] = args.lr
    if args.weight_decay:
        train_config["weight_decay"] = args.weight_decay
    if args.epochs:
        train_config["epochs"] = args.epochs
    if args.max_patience:
        train_config["max_patience"] = args.max_patience

    # Override experiment_config parameters from CLI if provided
    if args.report_per_period is not None:
        experiment_config["report_per_period"] = args.report_per_period
    if args.layers is not None:
        experiment_config["layers"] = parse_list(args.layers)
    experiment_config["export_results"] = args.export or \
            experiment_config.get("export_results", False)

    # Log configuration details
    print("Configuration Loaded:")
    print("  Model Config:")
    for key, value in model_config.items():
        print(f"    {key}: {value}")
    print("  Train Config:")
    for key, value in train_config.items():
        print(f"    {key}: {value}")
    print("  Experiment Config:")
    for key, value in experiment_config.items():
        print(f"    {key}: {value}")

    print(f"\nTraining and testing the model: {model_config['model_name']}...")
    model_config.update(train_config)  # combine configs
    train_and_test(
        params=model_config,
        layers=experiment_config.get("layers", TRAIN_LAYERS),
        report_per_period=experiment_config["report_per_period"],
        export_results=experiment_config["export_results"],
        print_results=True
    )

if __name__ == "__main__":
    main()