import argparse
import cv2
import torch
from ultralytics import YOLO # Assuming use of Ultralytics YOLO
import time
from tqdm import tqdm

def infer_on_video(video_path, model_path, output_path, conf_thresh=0.25, iou_thresh=0.45, model_type='yolo_detector'):
    """
    Runs detection or segmentation inference on a video and saves the output.

    Args:
        video_path (str): Path to the input video file.
        model_path (str): Path to the trained model weights (.pt file).
        output_path (str): Path to save the output video with detections.
        conf_thresh (float): Confidence threshold for detections.
        iou_thresh (float): IoU threshold for NMS.
        model_type (str): 'yolo_detector' for bounding boxes,
                          'canopy_segmenter' for segmentation masks.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    try:
        model = YOLO(model_path)
        model.to(device)
        print(f"Loaded {model_type} model from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define the codec and create VideoWriter object
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Or 'XVID' for .avi
    out_video = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    print(f"Processing video: {video_path}")
    print(f"Output will be saved to: {output_path}")

    frame_count = 0
    start_time = time.time()

    with tqdm(total=total_frames, desc="Inferring on video") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1

            # Perform inference
            # For Ultralytics YOLO: results is a list (usually 1 element for 1 image)
            # Each result object has boxes, masks, probs etc.
            results = model.predict(frame, conf=conf_thresh, iou=iou_thresh, device=device, verbose=False)
            
            # Annotate frame using results.plot() from Ultralytics
            annotated_frame = results[0].plot() # plot() draws bboxes, masks, labels

            # # Manual plotting if needed:
            # annotated_frame = frame.copy()
            # if results[0].boxes:
            #     for box in results[0].boxes.data.cpu().numpy(): # x1, y1, x2, y2, conf, class_id
            #         x1, y1, x2, y2, conf, cls_id = box
            #         cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            #         label = f"{model.names[int(cls_id)]} {conf:.2f}"
            #         cv2.putText(annotated_frame, label, (int(x1), int(y1)-10), 
            #                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
            # if model_type == 'canopy_segmenter' and results[0].masks:
            #     # Plot masks (this can be complex, results[0].plot() handles it)
            #     # For example, overlaying masks:
            #     for mask_data in results[0].masks.data.cpu().numpy(): # (H, W) binary mask
            #         # Resize mask_data if necessary to frame size
            #         resized_mask = cv2.resize(mask_data, (frame_width, frame_height))
            #         colored_mask = np.zeros_like(annotated_frame, dtype=np.uint8)
            #         colored_mask[resized_mask > 0] = [0,0,255] # Example: Red mask
            #         annotated_frame = cv2.addWeighted(annotated_frame, 1.0, colored_mask, 0.5, 0)


            out_video.write(annotated_frame)
            pbar.update(1)

            # cv2.imshow('Inference', annotated_frame) # Optional: display live
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
    
    end_time = time.time()
    processing_time = end_time - start_time
    avg_fps = frame_count / processing_time if processing_time > 0 else 0

    print(f"\nFinished processing video.")
    print(f"Total frames: {frame_count}")
    print(f"Processing time: {processing_time:.2f} seconds")
    print(f"Average FPS: {avg_fps:.2f}")

    cap.release()
    out_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run YOLO detection/segmentation inference on a video.")
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained YOLO model (.pt file).")
    parser.add_argument('--output_path', type=str, required=True, help="Path to save the output video with detections.")
    parser.add_argument('--conf_thresh', type=float, default=0.25, help="Confidence threshold for detections.")
    parser.add_argument('--iou_thresh', type=float, default=0.45, help="IoU threshold for NMS.")
    parser.add_argument('--model_type', type=str, default='yolo_detector', choices=['yolo_detector', 'canopy_segmenter'],
                        help="Type of model: 'yolo_detector' or 'canopy_segmenter'.")
    args = parser.parse_args()

    infer_on_video(args.video_path, args.model_path, args.output_path,
                   args.conf_thresh, args.iou_thresh, args.model_type)