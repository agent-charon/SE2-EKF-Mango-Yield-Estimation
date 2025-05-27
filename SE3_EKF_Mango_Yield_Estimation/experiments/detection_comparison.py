import yaml
import os
import subprocess # To call training/validation scripts
import pandas as pd # To tabulate results
# Assume a utility for calculating mAP if not directly from YOLO validation output
# from ..utils.metrics import calculate_map_etc (example utility)

# This script would simulate running training and validation for different detection models
# as described in Table I of the paper (MangoNet, YOLOv5s, MangoYOLO5, YOLOv8s).
# For this skeleton, we'll assume YOLOv8s is the focus and others are for comparison.
# A real comparison would require implementations or access to those other models.

def run_detection_experiment(model_name, train_script, val_script, 
                             training_config_template, dataset_config_yolo, 
                             model_arch_config_path):
    """
    Simulates training and validating a single detection model.
    Returns a dictionary of metrics.
    """
    print(f"\n--- Running Experiment for Detector: {model_name} ---")
    
    # 1. Prepare model-specific training config (if necessary)
    #    e.g., modify training_config_template for this model_name
    #    This might involve setting different base models or parameters.
    #    For simplicity, we assume training.yaml can handle different model types
    #    by just changing the `model_type` or specific model paths.

    # 2. Train the model (simulated - in reality, this calls train_detection.py)
    #    `python scripts/train_detection.py --config_path <specific_training_cfg> --model_type <model_name_yolo_type>`
    #    The path to best.pt would be captured from training output.
    print(f"Simulating training for {model_name}...")
    # trained_model_path = f"outputs/checkpoints/{model_name}/best.pt" # Example path
    # For this skeleton, let's assume models are pre-trained or training is separate.
    # We'll focus on how validation might be called.
    
    # For actual run:
    # cmd_train = [
    #     "python", train_script,
    #     "--config_path", "configs/training.yaml", # Assuming this can be adapted
    #     "--model_type", "yolo_detector" # And model.yaml points to correct arch for `model_name`
    # ]
    # subprocess.run(cmd_train, check=True)
    # trained_model_path = ... # find best.pt

    # For now, assume a placeholder path for a pre-trained model or skip validation if no path
    trained_model_path = f"models_for_comparison/{model_name}_weights.pt" # Placeholder
    if not os.path.exists(trained_model_path) and model_name == "YOLOv8s": # Our main model
        print(f"Warning: Trained model for {model_name} not found at {trained_model_path}. Validation will be skipped.")
        print("Please train YOLOv8s first using scripts/train_detection.py")
        # Or point to the actual trained model path from your training run.
        # e.g. outputs/checkpoints/yolo_detector/yolo_detector_training_run/weights/best.pt
        trained_model_path = "outputs/checkpoints/yolo_detector/yolo_detector_training_run/weights/best.pt" # Try default
        if not os.path.exists(trained_model_path):
             print(f"Still not found at default path. Skipping validation for {model_name}")
             return {"model": model_name, "mAP (%)": "N/A", "Precision (%)": "N/A", "Recall (%)": "N/A", "F1 score": "N/A", "Estimation time": "N/A"}


    # 3. Validate the model (simulated - calls validate_detection.py)
    #    `python scripts/validate_detection.py --model_path <trained_model_path> --config_path <dataset_config_yolo> ...`
    #    This would parse mAP, Precision, Recall from validation output.
    print(f"Simulating validation for {model_name} using {trained_model_path}...")
    # For actual run:
    # cmd_val = [
    #     "python", val_script,
    #     "--model_path", trained_model_path,
    #     "--config_path", dataset_config_yolo, # This is the data.yaml for YOLO
    #     "--model_type", "yolo_detector"
    # ]
    # result = subprocess.run(cmd_val, capture_output=True, text=True, check=True)
    # metrics = parse_validation_output(result.stdout) # Helper to parse text output

    # For this skeleton, let's use placeholder metrics similar to Table I
    # A real script would get these from model.val() if using Ultralytics directly
    # or by parsing stdout from validate_detection.py
    if model_name == "YOLOv8s":
        # Simulate running our main YOLOv8s
        # This part would ideally call the validate_model function from scripts.validate_detection
        # but that prints to console. We need to capture its returned metrics.
        # For now, direct call for concept (if validate_model is refactored to return metrics)
        try:
            from scripts.validate_detection import YOLO as ValYOLO # Use YOLO from ultralytics
            model_val = ValYOLO(trained_model_path)
            metrics_obj = model_val.val(data=dataset_config_yolo, imgsz=640, batch=16, verbose=False)
            # These are example attributes from Ultralytics metrics object
            precision = metrics_obj.box.mp * 100 if metrics_obj.box.mp else 0.0
            recall = metrics_obj.box.mr * 100 if metrics_obj.box.mr else 0.0
            map50 = metrics_obj.box.map50 * 100 if metrics_obj.box.map50 else 0.0 # mAP@0.5
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            # Paper's mAP seems to be mAP@0.5 as well, given YOLO contexts.
            # Estimation time is harder to get programmatically without specific benchmarking.
            metrics = {
                "model": model_name,
                "Precision (%)": f"{precision:.1f}",
                "Recall (%)": f"{recall:.1f}",
                "mAP (%)": f"{map50:.1f}", # Assuming paper's mAP is mAP@0.5
                "F1 score": f"{f1/100:.3f}", # F1 often 0-1 scale
                "Estimation time": "~30 min" # From paper
            }
        except Exception as e:
            print(f"Could not run validation for YOLOv8s: {e}")
            metrics = {"model": model_name, "mAP (%)": "Error", "Precision (%)": "Error", "Recall (%)": "Error", "F1 score": "Error", "Estimation time": "N/A"}

    elif model_name == "MangoNet":
        metrics = {"model": model_name, "Precision (%)": 78.8, "Recall (%)": 55.0, "mAP (%)": 63.8, "F1 score": 0.647, "Estimation time": "~360 min"}
    elif model_name == "YOLOv5s":
        metrics = {"model": model_name, "Precision (%)": 70.8, "Recall (%)": 64.6, "mAP (%)": 70.8, "F1 score": 0.675, "Estimation time": "~75 min"}
    elif model_name == "MangoYOLO5":
        metrics = {"model": model_name, "Precision (%)": 88.3, "Recall (%)": 74.9, "mAP (%)": 85.1, "F1 score": 0.813, "Estimation time": "~42 min"}
    else:
        metrics = {"model": model_name, "mAP (%)": "N/A", "Precision (%)": "N/A", "Recall (%)": "N/A", "F1 score": "N/A", "Estimation time": "N/A"}
    
    return metrics


def main():
    print("--- Running Detection Model Comparison (Table I Reproduction) ---")
    
    # Paths to scripts and base configs
    train_script_path = "scripts/train_detection.py"
    val_script_path = "scripts/validate_detection.py"
    
    # This needs to be the YOLO-formatted data.yaml
    # Ensure it's correctly set up in your configs.
    # For example, create a 'configs/mango_detector_data_yoloformat.yaml'
    # and point dataset_config_for_yolo to it.
    dataset_config_for_yolo = "configs/dataset_yolo_format.yaml" # Placeholder for actual YOLO data YAML

    if not os.path.exists(dataset_config_for_yolo):
        print(f"Error: Dataset config for YOLO ({dataset_config_for_yolo}) not found.")
        print("Please create a YAML file (e.g., data/mango_yolo_data.yaml) with train/val paths and class info,")
        print("then update the path in this script.")
        return

    model_names = ["MangoNet", "YOLOv5s", "MangoYOLO5", "YOLOv8s"] # As per Table I
    all_results = []

    for model_name in model_names:
        # For MangoNet, YOLOv5s, MangoYOLO5, you'd need their specific training/validation setups
        # or use pre-existing results if just tabulating.
        # The run_detection_experiment would need to be adapted.
        # For this skeleton, it mostly simulates for others and tries to run for YOLOv8s.
        
        # Path to model architecture config (if building from components, not used if loading .pt)
        model_arch_cfg = "configs/model.yaml" # Assumes this can define different types
        
        result = run_detection_experiment(
            model_name,
            train_script_path,
            val_script_path,
            "configs/training.yaml", # Base training config template
            dataset_config_for_yolo,
            model_arch_cfg
        )
        all_results.append(result)

    # Display results in a table
    results_df = pd.DataFrame(all_results)
    print("\n--- Detection Comparison Results (Simulated Table I) ---")
    print(results_df.to_string(index=False))

    # Save to CSV
    results_df.to_csv("outputs/results/detection_comparison_table_I.csv", index=False)
    print("\nResults saved to outputs/results/detection_comparison_table_I.csv")

if __name__ == '__main__':
    # This script requires that:
    # 1. `scripts/train_detection.py` and `scripts/validate_detection.py` are runnable.
    # 2. A YOLO-formatted dataset YAML exists (e.g., `configs/dataset_yolo_format.yaml`).
    # 3. For non-YOLOv8s models, their specific training/evaluation logic would need to be integrated
    #    or results manually entered. This skeleton focuses on showing how one might structure
    #    the experiment for the main YOLOv8s model.
    
    # Create a dummy dataset_yolo_format.yaml for the script to run without error
    # In a real scenario, this file MUST be correctly configured.
    dummy_yolo_data_yaml_content = """
train: ../data/dummy_yolo/images/train
val: ../data/dummy_yolo/images/val
nc: 1
names: ['mango']
    """
    dummy_yolo_data_yaml_path = "configs/dataset_yolo_format.yaml"
    if not os.path.exists(dummy_yolo_data_yaml_path):
        os.makedirs(os.path.dirname(dummy_yolo_data_yaml_path), exist_ok=True)
        with open(dummy_yolo_data_yaml_path, "w") as f:
            f.write(dummy_yolo_data_yaml_content)
        print(f"Created dummy {dummy_yolo_data_yaml_path} for testing experiment script.")
        # Create dummy model file for YOLOv8s to simulate it exists
        os.makedirs("models_for_comparison", exist_ok=True)
        if not os.path.exists("models_for_comparison/YOLOv8s_weights.pt"):
             if not os.path.exists("outputs/checkpoints/yolo_detector/yolo_detector_training_run/weights/best.pt"):
                # Create a dummy file if no actual trained model exists
                # This is just to allow the script to run without FileNotFoundError for this placeholder path
                # In a real run, this should be the path to an actual trained model.
                dummy_model_path = "outputs/checkpoints/yolo_detector/yolo_detector_training_run/weights/best.pt"
                os.makedirs(os.path.dirname(dummy_model_path), exist_ok=True)
                with open(dummy_model_path, "w") as f: f.write("dummy weights")
                print(f"Created dummy model file at {dummy_model_path} for YOLOv8s experiment.")


    main()