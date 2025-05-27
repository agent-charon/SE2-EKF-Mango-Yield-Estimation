import argparse
import yaml
import torch
from ultralytics import YOLO # Assuming use of Ultralytics YOLO

def validate_model(model_path, config_path, model_type):
    """
    Validates a trained detection or segmentation model.

    Args:
        model_path (str): Path to the trained model weights (.pt file).
        config_path (str): Path to the dataset configuration YAML file
                           (formatted for Ultralytics YOLO, specifying val path, nc, names).
        model_type (str): 'yolo_detector' or 'canopy_segmenter'.
    """
    try:
        with open(config_path, 'r') as f:
            dataset_cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Dataset configuration file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading dataset configuration: {e}")
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Ultralytics val() function expects certain parameters like 'data' (path to dataset yaml)
    # and other validation settings.
    # The 'data' yaml should point to the validation set.

    print(f"--- Validating {model_type} ---")
    print(f"Model Path: {model_path}")
    print(f"Dataset Config (for YOLO lib validation set): {config_path}")
    print(f"Device: {device}")

    try:
        # Initialize YOLO model from the trained weights
        model = YOLO(model_path)
        model.to(device)

        # Start validation
        # Refer to Ultralytics documentation for full validation parameters.
        # Key parameters:
        # data: path to the dataset_config.yaml which specifies 'val' path.
        # imgsz: image size used for validation (should match training).
        # batch: batch size for validation.
        # conf: confidence threshold for NMS.
        # iou: IoU threshold for NMS and for mAP calculation.
        # split: 'val' or 'test' (must be defined in data yaml).

        # Extract some common validation params that might be in dataset_cfg or use defaults
        imgsz = dataset_cfg.get('imgsz', 640) # Default if not in data yaml
        batch_size_val = dataset_cfg.get('batch_size_val', 16) # Default

        print(f"Using ImgSz: {imgsz}, Batch Size: {batch_size_val} for validation.")

        metrics = model.val(
            data=config_path, # This YAML must define the 'val' path.
            imgsz=imgsz,
            batch=batch_size_val,
            device=device,
            conf=0.25, # Default confidence for NMS during val, can be tuned
            iou=0.5,   # Default IoU for mAP@0.5
            # split='val', # Specify if your data yaml has multiple splits
            # project="outputs/validation_results/", # Where to save detailed plots/results
            # name=f"{model_type}_validation"
            verbose=True # Show detailed output
        )
        
        print(f"Validation completed for {model_type}.")
        
        if model_type == 'yolo_detector':
            print(f"  mAP@0.5-0.95: {metrics.box.map:.4f}")
            print(f"  mAP@0.5: {metrics.box.map50:.4f}")
            print(f"  Precision: {metrics.box.mp:.4f}") # Mean Precision
            print(f"  Recall: {metrics.box.mr:.4f}")    # Mean Recall
        elif model_type == 'canopy_segmenter':
            # For segmentation, metrics are different (e.g., mask mAP)
            print(f"  Mask mAP@0.5-0.95: {metrics.seg.map:.4f}")
            print(f"  Mask mAP@0.5: {metrics.seg.map50:.4f}")
            # Add other relevant segmentation metrics if available from `metrics.seg`

        # metrics object contains more detailed results.
        # print(metrics)

    except ImportError:
        print("Error: Ultralytics YOLO library not found. Please install it to run validation.")
    except FileNotFoundError as e:
        print(f"Error during validation setup (file not found): {e}")
        print("Ensure model path and dataset YAML path are correct and files exist.")
    except Exception as e:
        print(f"An error occurred during validation of {model_type}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Validate Detection/Segmentation Model")
    parser.add_argument('--model_path', type=str, required=True,
                        help="Path to the trained model weights (.pt file, e.g., outputs/checkpoints/yolo_detector/train/weights/best.pt)")
    parser.add_argument('--config_path', type=str, required=True,
                        help="Path to the dataset YAML configuration file (formatted for Ultralytics, pointing to val set, e.g., configs/dataset_yolo_format.yaml)")
    parser.add_argument('--model_type', type=str, required=True, choices=['yolo_detector', 'canopy_segmenter'],
                        help="Type of model to validate: 'yolo_detector' or 'canopy_segmenter'")
    args = parser.parse_args()

    print("Reminder: Ensure your dataset YAML (config_path argument) is formatted correctly for Ultralytics YOLO,")
    print("and specifically defines the 'val' path for the validation dataset, 'nc', and 'names'.")

    validate_model(args.model_path, args.config_path, args.model_type)