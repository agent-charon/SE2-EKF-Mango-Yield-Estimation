import cv2
import time
import os
from datetime import datetime

def capture_video_360(output_dir="data/raw_videos/new_captures", duration_seconds=90, camera_index=0, tree_id="unknown_tree"):
    """
    Captures video from a camera, simulating a 360-degree capture by manual movement.

    Args:
        output_dir (str): Directory to save the captured video.
        duration_seconds (int): Desired duration of the video capture.
        camera_index (int): Index of the camera to use.
        tree_id (str): Identifier for the tree being captured.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print(f"Error: Could not open camera with index {camera_index}.")
        return

    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: # Fallback if fps is not reported
        fps = 30
        print(f"Warning: Camera FPS reported as 0, defaulting to {fps}.")


    # Define the codec and create VideoWriter object
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = os.path.join(output_dir, f"tree_{tree_id}_capture_{timestamp}.mp4") # Using MP4
    # Common codecs: 'XVID' for .avi, 'mp4v' or 'avc1' for .mp4
    # Check fourcc.org for more. 'mp4v' is generally compatible.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    print(f"Starting video capture for {duration_seconds} seconds.")
    print(f"Press 'q' to stop early. Video will be saved to: {video_filename}")
    print("Please move the camera slowly around the tree to simulate a 360-degree view.")

    start_time = time.time()
    frames_written = 0

    try:
        while (time.time() - start_time) < duration_seconds:
            ret, frame = cap.read()
            if not ret:
                print("Error: Can't receive frame (stream end?). Exiting ...")
                break

            out.write(frame)
            frames_written += 1

            cv2.imshow('Video Capture - Press Q to Stop', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Capture stopped by user.")
                break
    finally:
        # Release everything if job is finished
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video capture finished. {frames_written} frames written to {video_filename}.")
        if frames_written == 0:
             print("Warning: No frames were written. Check camera connection and permissions.")


if __name__ == '__main__':
    # Example usage:
    tree_identifier = input("Enter Tree ID (e.g., Banganapalle_Tree_01): ")
    capture_duration = int(input("Enter capture duration in seconds (e.g., 90): "))
    
    # Specify the output directory or use the default
    custom_output_dir = "data/raw_videos/mango_orchard_session1" 
    # Ensure the directory exists, or let the function create its default
    if not os.path.exists(custom_output_dir) and custom_output_dir != "data/raw_videos/new_captures":
        os.makedirs(custom_output_dir)
        print(f"Created directory: {custom_output_dir}")
        
    capture_video_360(output_dir=custom_output_dir, duration_seconds=capture_duration, tree_id=tree_identifier)