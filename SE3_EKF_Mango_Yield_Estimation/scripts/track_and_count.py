import argparse
import cv2
import numpy as np
import yaml
import torch
from tqdm import tqdm
import time
import os

# Preprocessing imports
from preprocessing.image_enhancement.gamma_correction import GammaCorrection
from preprocessing.image_enhancement.gaussian_smoothing import GaussianSmoothing
from preprocessing.image_enhancement.hue_saturation_adjust import HueSaturationAdjust
from preprocessing.video_stabilization.motion_estimation import MotionEstimator
from preprocessing.video_stabilization.motion_smoothing import MotionSmoother
from preprocessing.video_stabilization.motion_compensation import MotionCompensator
from preprocessing.redundant_frame_removal.cosine_similarity_rfr import CosineSimilarityRFR
from preprocessing.canopy_segmentation.masked_yolo_segment import MaskedYOLOSegmenter

# Detection import (Ultralytics YOLO)
from ultralytics import YOLO

# Tracking imports
from tracking.ekf.id_assignment import IDAssigner
# (Other EKF components are used internally by IDAssigner and ExtendedKalmanFilter)


def load_configs(config_dir):
    configs = {}
    try:
        with open(os.path.join(config_dir, 'preprocessing.yaml'), 'r') as f:
            configs['preprocessing'] = yaml.safe_load(f)
        with open(os.path.join(config_dir, 'tracking.yaml'), 'r') as f: # For IDAssigner's EKF params
            configs['tracking_ekf'] = yaml.safe_load(f) 
        # model.yaml and dataset.yaml are usually for training/val, not directly needed here
        # unless specific parameters like num_classes are read.
    except FileNotFoundError as e:
        print(f"Error: Configuration file not found in {config_dir}. {e}")
        raise
    except Exception as e:
        print(f"Error loading configurations: {e}")
        raise
    return configs

def get_observed_angle_from_detection(detection_data, frame_shape):
    """
    Placeholder/Simple method to estimate an observed angle from a bounding box.
    The paper mentions "camera movements through angular estimation" in EKF.
    The source of this observed angle for the EKF update step needs to be defined.
    It could be:
    1. From an Oriented Bounding Box detector (if YOLOv8s can provide it or another model).
    2. Estimated from features within the bbox (e.g., main axis via moments, or ORB as hinted in Fig 7 description).
    3. Assumed to be zero or a fixed value if not reliably estimable per detection.
    4. Derived from global camera motion if tracking relative to a stable world frame.

    For this skeleton, let's return 0.0. A real implementation needs a robust angle source.
    Fig 7 shows "Homography Mapping of consecutive frames" and "ORB keyfeature detection"
    leading to "Change in Rotation Angle over Video Frames". This suggests a global or
    per-object angle estimation tied to frame-to-frame camera motion or object appearance.
    If it's global camera rotation, it might not be a per-detection attribute but rather
    an input to the EKF's interpretation of measurements.
    The EKF's measurement model (Hk) currently assumes `theta_obs` is directly measured.
    """
    # Example: if bbox implies orientation (e.g. from PCA of mask)
    # x1, y1, x2, y2 = detection_data[0] # bbox_xyxy
    # cx = (x1+x2)/2; cy = (y1+y2)/2
    # width = x2-x1; height = y2-y1
    # angle = 0.0 # Default
    # if width > height: angle = 0.0 # Horizontal
    # else: angle = np.pi / 2.0 # Vertical (this is very naive)
    return 0.0 # Placeholder

def main(video_path, detector_model_path, segmenter_model_path, output_path, config_dir):
    configs = load_configs(config_dir)
    prep_cfg = configs['preprocessing']
    # tracking_ekf_cfg is implicitly used by IDAssigner when creating EKF instances

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # --- Initialize Preprocessing Modules ---
    gamma_corrector = GammaCorrection(gamma=prep_cfg['image_enhancement']['gamma_correction']['gamma'])
    gaussian_smoother = GaussianSmoothing(
        kernel_size=tuple(prep_cfg['image_enhancement']['gaussian_smoothing']['kernel_size']),
        sigma_x=prep_cfg['image_enhancement']['gaussian_smoothing']['sigma_x']
    )
    # HueSaturationAdjuster might be used for augmentation during training,
    # For inference, usually fixed enhancements or none. Let's assume it's not applied per frame here unless specified.

    # Video Stabilizer components
    motion_estimator = MotionEstimator(
        max_corners=prep_cfg['video_stabilization']['motion_estimation'].get('max_corners', 100),
        quality_level=prep_cfg['video_stabilization']['motion_estimation'].get('quality_level', 0.01),
        min_distance=prep_cfg['video_stabilization']['motion_estimation'].get('min_distance', 10)
    )
    motion_smoother = MotionSmoother(
        smoothing_radius=prep_cfg['video_stabilization']['motion_smoothing'].get('smoothing_radius', 15)
    )
    motion_compensator = MotionCompensator()
    
    # Store all estimated raw transforms (T_k) and then all smoothed transforms (T_smooth_k)
    # to calculate compensation M_comp = T_smooth_k * inv(T_k)
    # This requires a two-pass approach for stabilization or a sufficiently large buffer.
    # For simplicity in a single pass, we'll apply smoothing to a sliding window of transforms
    # and derive compensation based on that. The paper's description is high-level.
    # A simpler 1-pass approach for smoothing:
    #   T_k = estimate(frame_k, frame_{k-1})
    #   smoothed_T_k = motion_smoother.smooth_current_transform(T_k) # Needs modification to MotionSmoother
    #   M_comp = smoothed_T_k @ np.linalg.inv(T_k_as_3x3)
    # For now, let's process stabilization in a somewhat offline manner for this script example,
    # assuming we collect all transforms first, then smooth, then apply.
    # This means the script might need to be structured for two passes over video data for full stabilization.
    # For a single pass approximation:
    # We get T_k, add it to smoother's history. Smoother gives smoothed (dx,dy,da) for current step.
    # Construct M_raw from T_k, M_smooth from smoothed (dx,dy,da). M_comp = M_smooth * inv(M_raw).
    
    # Redundant Frame Remover
    rfr = CosineSimilarityRFR(
        similarity_threshold=prep_cfg['redundant_frame_removal']['cosine_similarity_threshold']
    )
    
    # Canopy Segmenter
    try:
        canopy_segmenter = MaskedYOLOSegmenter(model_weights_path=segmenter_model_path, device=device)
        print("Canopy segmenter loaded.")
    except Exception as e:
        print(f"Could not load canopy segmenter: {e}. Proceeding without canopy segmentation.")
        canopy_segmenter = None
        
    # --- Initialize Detector ---
    try:
        detector = YOLO(detector_model_path)
        detector.to(device)
        print("Mango detector loaded.")
    except Exception as e:
        print(f"Fatal: Could not load mango detector: {e}")
        return

    # --- Initialize Tracker ---
    # IDAssigner loads EKF config from the path given in its constructor
    assigner = IDAssigner(
        iou_threshold=configs['tracking_ekf']['ekf'].get('id_assignment', {}).get('iou_threshold', 0.3),
        max_age=configs['tracking_ekf']['ekf'].get('id_assignment', {}).get('max_age', 30),
        min_hits_to_confirm=configs['tracking_ekf']['ekf'].get('id_assignment', {}).get('min_hits', 3),
        ekf_config_path=os.path.join(config_dir, 'tracking.yaml')
    )
    print("ID Assigner (with EKF) initialized.")

    # --- Video Processing ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30 # Fallback
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video '{video_path}' for tracking and counting...")
    
    # For stabilization (simplified single pass attempt)
    prev_gray_stabilize = None
    accumulated_raw_transforms = [] # Store sequence of T_k (raw inter-frame)
    stabilization_transforms_to_apply = [] # Store M_compensation for each frame

    # --- Pass 1 (Optional but better for stabilization): Collect Transforms ---
    # If doing full stabilization, this pass collects all T_k.
    # Then, smooth the whole sequence of (dx,dy,da) derived from T_k.
    # Then, compute M_compensation for each frame.
    # For this script, let's try a simpler online smoothing for stabilization.
    
    frame_idx = 0
    # Unique mango IDs observed in the video
    observed_mango_ids = set()

    with tqdm(total=total_frames, desc="Tracking Mangoes") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            processed_frame = frame.copy()

            # 1. Redundant Frame Removal (optional, could be first)
            # If RFR is applied, subsequent processing only happens on kept frames.
            # The EKF dt should correspond to time between *kept* frames.
            # This complicates dt if RFR is aggressive. Paper's dt=1s is for original 30fps.
            # Let's assume RFR is light or EKF dt is adaptive (not in this skeleton).
            # For simplicity, let's apply RFR check but process all frames for video output,
            # and only feed non-redundant to detector/tracker logic for counting.
            # Or, if frame is redundant, we skip detection/tracking for it.
            
            is_non_redundant_for_logic = rfr.check_frame(frame.copy()) # Use copy for RFR internal state

            # --- Preprocessing ---
            # 1.1 Image Enhancement
            processed_frame = gamma_corrector.apply(processed_frame)
            processed_frame = gaussian_smoother.apply(processed_frame)
            # Hue/Sat usually for training aug, skip in basic inference unless specifically beneficial

            # 1.2 Video Stabilization (Simplified Online)
            current_gray_stabilize = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            raw_affine_T_k = None
            if prev_gray_stabilize is not None:
                raw_affine_T_k = motion_estimator.estimate_transform(current_gray_stabilize)
            
            # Add to smoother (even if None, smoother handles it)
            motion_smoother.add_transform(raw_affine_T_k) # Smoother stores (dx,dy,da)
            
            # Get all smoothed motions so far
            all_smoothed_motions = motion_smoother.get_smoothed_trajectory()
            
            M_compensation = np.eye(2, 3, dtype=np.float32) # Default: no compensation
            if raw_affine_T_k is not None and all_smoothed_motions and len(all_smoothed_motions) > frame_idx :
                # Construct M_raw from raw_affine_T_k
                M_raw_3x3 = np.vstack([raw_affine_T_k, [0,0,1]])
                
                # Construct M_smooth from the latest smoothed (dx,dy,da)
                # This assumes get_smoothed_trajectory returns smoothed INTER-FRAME params
                sdx, sdy, sda = all_smoothed_motions[frame_idx] # Smoothed params for current transition
                M_smooth_k = np.array([
                    [np.cos(sda), -np.sin(sda), sdx],
                    [np.sin(sda),  np.cos(sda), sdy]
                ], dtype=np.float32)
                M_smooth_3x3 = np.vstack([M_smooth_k, [0,0,1]])
                
                try:
                    M_raw_inv_3x3 = np.linalg.inv(M_raw_3x3)
                    M_compensation_3x3 = M_smooth_3x3 @ M_raw_inv_3x3
                    M_compensation = M_compensation_3x3[:2, :]
                except np.linalg.LinAlgError:
                    # print("Warning: M_raw_3x3 is singular, cannot compute compensation transform.")
                    M_compensation = np.eye(2, 3, dtype=np.float32) # Fallback

            processed_frame = motion_compensator.compensate_frame(processed_frame, M_compensation)
            prev_gray_stabilize = current_gray_stabilize.copy()
            # End Stabilization


            # Frame to use for detection (after enhancements and stabilization)
            detection_input_frame = processed_frame.copy()

            # 1.3 Canopy Segmentation
            final_canopy_mask = None
            if canopy_segmenter:
                canopy_mask = canopy_segmenter.segment_canopy(detection_input_frame, 
                                                              confidence_threshold=prep_cfg['canopy_segmentation'].get('model_confidence_threshold', 0.5))
                if canopy_mask is not None:
                    # Apply mask to isolate the tree for detection
                    detection_input_frame = canopy_segmenter.apply_mask_to_frame(detection_input_frame, canopy_mask)
                    final_canopy_mask = canopy_mask # For visualization


            # --- Detection ---
            detections_for_tracker = [] # List of (bbox_xyxy, obs_theta, score, class_id)
            if is_non_redundant_for_logic or frame_idx % int(fps * configs['tracking_ekf']['ekf']['delta_t']) == 0 : # Process if non-redundant OR at EKF update interval
                # The paper uses dt=1sec for EKF, implying detection/tracking runs at 1Hz on the video stream.
                # If RFR makes intervals irregular, this needs careful thought for EKF's dt.
                # For now, let's assume EKF is robust or dt is average.
                # Or, only run detection/tracking logic IF non-redundant.
                
                # Run detector
                # Ultralytics predict returns a list of Results objects
                detector_results = detector.predict(detection_input_frame, conf=0.25, iou=0.45, device=device, verbose=False)

                if detector_results and detector_results[0].boxes:
                    for box_data in detector_results[0].boxes.data.cpu().numpy():
                        x1, y1, x2, y2, conf, cls_id = box_data
                        if int(cls_id) == 0: # Assuming mango is class 0
                            # Get observed angle (placeholder)
                            obs_angle = get_observed_angle_from_detection(box_data, frame.shape)
                            detections_for_tracker.append(
                                ([x1,y1,x2,y2], obs_angle, conf, int(cls_id))
                            )
            
            # --- Tracking ---
            active_tracks = assigner.process_detections(detections_for_tracker)
            
            # Update set of unique mango IDs seen so far
            for track_info in active_tracks:
                if track_info['status'] == 'confirmed': # Count only confirmed tracks
                     observed_mango_ids.add(track_info['id'])


            # --- Visualization on the original `processed_frame` (before canopy masking for detection input) ---
            output_display_frame = processed_frame.copy() # Start with stabilized & enhanced frame
            
            # Draw canopy mask if available (on the display frame)
            if final_canopy_mask is not None:
                 contours, _ = cv2.findContours(final_canopy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                 cv2.drawContours(output_display_frame, contours, -1, (0, 255, 255), 1) # Yellow contour for canopy

            # Draw tracks
            for track_info in active_tracks:
                tid = track_info['id']
                x1, y1, x2, y2 = map(int, track_info['bbox_xyxy'])
                # theta_rad = track_info['theta'] # Estimated orientation
                status = track_info['status']
                color = (0, 255, 0) if status == 'confirmed' else (255, 165, 0) # Green for confirmed, Orange for tentative
                
                cv2.rectangle(output_display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(output_display_frame, f"ID:{tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Could also draw oriented box using theta_rad if needed

            # Display current unique mango count
            cv2.putText(output_display_frame, f"Unique Mangoes: {len(observed_mango_ids)}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            out_video.write(output_display_frame)
            pbar.update(1)
            frame_idx += 1

            # if cv2.waitKey(1) & 0xFF == ord('q'): # For debugging
            #     break
    
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    print(f"\nProcessing complete. Output video saved to {output_path}")
    print(f"Final estimated unique mango count: {len(observed_mango_ids)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Mango Detection, Tracking, and Counting")
    parser.add_argument('--video_path', type=str, required=True, help="Path to input video.")
    parser.add_argument('--detector_model_path', type=str, required=True, help="Path to trained mango detector model (.pt).")
    parser.add_argument('--segmenter_model_path', type=str, required=True, help="Path to trained canopy segmenter model (.pt). Set to 'none' to skip.")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save output video with tracking.")
    parser.add_argument('--config_dir', type=str, default="configs/", help="Directory containing preprocessing.yaml and tracking.yaml.")
    
    args = parser.parse_args()

    if args.segmenter_model_path.lower() == 'none':
        segmenter_path = None
    else:
        segmenter_path = args.segmenter_model_path

    main(args.video_path, args.detector_model_path, segmenter_path, args.output_path, args.config_dir)