import argparse
import yaml
import os
import torch
from ultralytics import YOLO # Assuming use of Ultralytics YOLO for simplicity

def train_model(config_path, model_type):
    """
    Trains a detection or segmentation model.

    Args:
        config_path (str): Path to the main training configuration YAML file.
        model_type (str): 'yolo_detector' or 'canopy_segmenter'.
    """
    try:
        with open(config_path, 'r') as f:
            train_cfg_main = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Training configuration file not found at {config_path}")
        return
    except Exception as e:
        print(f"Error loading training configuration: {e}")
        return

    device = train_cfg_main.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    epochs = train_cfg_main.get('epochs', 100)
    batch_size = train_cfg_main.get('batch_size', 16)
    learning_rate = train_cfg_main.get('learning_rate', 0.01)
    optimizer_name = train_cfg_main.get('optimizer', 'AdamW')
    
    specific_train_cfg = {}
    model_yaml_path = ""
    dataset_yaml_path_for_yolo = "" # YOLO library expects a specific dataset YAML
    checkpoint_dir = ""
    model_variant_name = "" # e.g., 'yolov8s.pt' or 'yolov8s-seg.pt'

    if model_type == 'yolo_detector':
        specific_train_cfg = train_cfg_main.get('yolo_detector_training', {})
        model_config_file = specific_train_cfg.get('model_config', 'configs/model.yaml')
        dataset_config_file = specific_train_cfg.get('dataset_config', 'configs/dataset.yaml')
        checkpoint_dir = specific_train_cfg.get('checkpoint_dir', 'outputs/checkpoints/yolo_detector/')
        
        with open(model_config_file, 'r') as f_model:
            model_arch_cfg = yaml.safe_load(f_model)['yolo_detector']
        # For Ultralytics, model can be 'yolov8s.yaml' (arch) or 'yolov8s.pt' (pretrained)
        model_variant_name = model_arch_cfg.get('type', 'yolov8s') + ".yaml" # Build from arch
        if model_arch_cfg.get('pretrained_weights_path'): # Or load pretrained
             model_variant_name = model_arch_cfg.get('pretrained_weights_path', 'yolov8s.pt')

        # Ultralytics YOLO needs a data.yaml with train/val paths and class names
        # We need to construct this or ensure `dataset_config_file` IS this data.yaml
        # For this script, let's assume dataset_config_file (configs/dataset.yaml) needs to be parsed
        # to create the data.yaml that Ultralytics YOLO expects, or point directly to it.
        # Let's assume `dataset_config_file` is THE `data.yaml` for YOLO.
        with open(dataset_config_file, 'r') as f_data:
            data_paths_cfg = yaml.safe_load(f_data)
            # Ultralytics trainer needs paths to train/val image dirs and class names
            # This data_yaml should be formatted like:
            # train: ../data/processed_frames/train/images/
            # val: ../data/processed_frames/val/images/
            # nc: 1
            # names: ['mango']
            # For simplicity, we assume configs/dataset.yaml IS this data_yaml or can be referenced.
            dataset_yaml_path_for_yolo = dataset_config_file # Crucial: This must be formatted for YOLO lib

    elif model_type == 'canopy_segmenter':
        specific_train_cfg = train_cfg_main.get('canopy_segmenter_training', {})
        model_config_file = specific_train_cfg.get('model_config', 'configs/model.yaml')
        dataset_config_file = specific_train_cfg.get('dataset_config', 'configs/dataset.yaml')
        checkpoint_dir = specific_train_cfg.get('checkpoint_dir', 'outputs/checkpoints/canopy_segmenter/')

        with open(model_config_file, 'r') as f_model:
            model_arch_cfg = yaml.safe_load(f_model)['canopy_segmenter']
        # For segmentation, e.g., 'yolov8s-seg.yaml' or 'yolov8s-seg.pt'
        model_variant_name = model_arch_cfg.get('type', 'yolov8s-seg') + ".yaml"
        if model_arch_cfg.get('pretrained_weights_path'):
            model_variant_name = model_arch_cfg.get('pretrained_weights_path', 'yolov8s-seg.pt')
        
        # Similar to detector, segmentation training needs a data.yaml
        dataset_yaml_path_for_yolo = dataset_config_file # Assume it's formatted for YOLO seg

    else:
        print(f"Error: Unknown model_type '{model_type}'")
        return

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    print(f"--- Training {model_type} ---")
    print(f"Model Variant: {model_variant_name}")
    print(f"Dataset Config (for YOLO lib): {dataset_yaml_path_for_yolo}")
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, LR: {learning_rate}, Optimizer: {optimizer_name}")
    print(f"Device: {device}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}")

    try:
        # Initialize YOLO model from Ultralytics
        # model_variant_name could be 'yolov8s.yaml' (to build) or 'yolov8s.pt' (to load and fine-tune)
        model = YOLO(model_variant_name)
        model.to(device)

        # Augmentation parameters (can be passed directly to train if supported)
        augs = specific_train_cfg.get('augmentations', {})

        # Start training
        # Note: Ultralytics train() has many more parameters. This is a simplified call.
        # Refer to Ultralytics documentation for full control.
        model.train(
            data=dataset_yaml_path_for_yolo,
            epochs=epochs,
            batch=batch_size,
            imgsz=train_cfg_main.get('imgsz', 640), # Input image size
            device=device,
            optimizer=optimizer_name,
            lr0=learning_rate, # Initial learning rate
            # patience=specific_train_cfg.get('patience', 50), # Early stopping patience
            # workers=specific_train_cfg.get('workers', 8),
            project=checkpoint_dir, # Saves results in checkpoint_dir/train, checkpoint_dir/train2 etc.
            name=f'{model_type}_training_run',
            # Pass augmentation parameters directly if supported by the version of ultralytics
            # Otherwise, they are often part of the dataset augmentation pipeline referenced in data_yaml
            # Example for some common augs:
            hsv_h=augs.get('hsv_h', 0.015),
            hsv_s=augs.get('hsv_s', 0.7),
            hsv_v=augs.get('hsv_v', 0.4),
            degrees=augs.get('degrees', 0.0),
            translate=augs.get('translate', 0.1),
            scale=augs.get('scale', 0.5),
            fliplr=augs.get('fliplr', 0.5),
            mosaic=augs.get('mosaic', 1.0) # Mosaic is often controlled by data loader
        )
        
        print(f"Training completed for {model_type}. Model weights saved in {checkpoint_dir}.")
        # The best model is often saved as 'best.pt' in the run directory.

    except ImportError:
        print("Error: Ultralytics YOLO library not found. Please install it to run training.")
    except FileNotFoundError as e:
        print(f"Error during training setup (file not found): {e}")
        print("Ensure model variant name and dataset YAML path are correct and files exist.")
    except Exception as e:
        print(f"An error occurred during training {model_type}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Detection/Segmentation Model")
    parser.add_argument('--config_path', type=str, required=True, help="Path to the main training YAML configuration file (e.g., configs/training.yaml)")
    parser.add_argument('--model_type', type=str, required=True, choices=['yolo_detector', 'canopy_segmenter'],
                        help="Type of model to train: 'yolo_detector' or 'canopy_segmenter'")
    args = parser.parse_args()

    # Crucial Setup for Ultralytics:
    # The `configs/dataset.yaml` needs to be structured like a YOLO `data.yaml`, e.g.:
    # ```yaml
    # # configs/dataset_yolo_format.yaml
    # path: ../data/yolo_dataset # Root directory of dataset
    # train: images/train  # train images (relative to 'path')
    # val: images/val    # val images (relative to 'path')
    # test: images/test # Optional
    # 
    # names:
    #   0: mango # For detector
    #   # For segmenter, if class is 'canopy'
    #   # 0: canopy
    # ```
    # The paths in `configs/dataset.yaml` (`train_image_dir`, `canopy_train_image_dir` etc.)
    # from my earlier generation need to be adapted into this YOLO format data.yaml.
    # The current script assumes `specific_train_cfg.get('dataset_config')` IS this path.

    print("Reminder: Ensure your dataset YAML (referenced in training.yaml -> model_type_training -> dataset_config) ")
    print("is formatted correctly for the Ultralytics YOLO library (specifying train/val paths and class names).")
    print("The paths inside that YAML should be relative to the YAML's location or absolute.")

    train_model(args.config_path, args.model_type)