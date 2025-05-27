import cv2
import numpy as np

class MotionEstimator:
    def __init__(self, max_corners=100, quality_level=0.01, min_distance=10, block_size=7):
        """
        Estimates motion (affine transformation) between two frames using feature tracking.

        Args:
            max_corners (int): Maximum number of corners to detect using Shi-Tomasi.
            quality_level (float): Quality level for Shi-Tomasi corner detection.
            min_distance (int): Minimum possible Euclidean distance between corners.
            block_size (int): Size of an average block for computing a derivative covariation matrix over each pixel neighborhood.
        """
        self.feature_params = dict(maxCorners=max_corners,
                                   qualityLevel=quality_level,
                                   minDistance=min_distance,
                                   blockSize=block_size)
        self.lk_params = dict(winSize=(15, 15), # Lucas-Kanade window size
                              maxLevel=2,      # Max pyramid levels
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        self.prev_gray = None
        self.prev_points = None

    def estimate_transform(self, current_frame_gray):
        """
        Estimates the affine transformation from the previous frame to the current frame.

        Args:
            current_frame_gray (np.array): Grayscale current frame.

        Returns:
            np.array or None: 2x3 Affine transformation matrix, or None if estimation fails.
        """
        if self.prev_gray is None:
            self.prev_gray = current_frame_gray.copy()
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.prev_points is None:
                # print("Warning: No features found in the first frame for motion estimation.")
                return None # Cannot estimate transform for the very first frame or if no features
            return np.eye(2, 3, dtype=np.float32) # Return identity for the first frame

        if self.prev_points is None or len(self.prev_points) < 4: # Need at least 3 points for affine, 4 for robustness
            # print("Warning: Not enough previous points to track. Re-detecting.")
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            if self.prev_points is None or len(self.prev_points) < 4:
                # print("Warning: Still not enough features after re-detection.")
                self.prev_gray = current_frame_gray.copy() # Update prev_gray for next attempt
                return None


        # Calculate optical flow
        current_points, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, current_frame_gray,
                                                               self.prev_points, None, **self.lk_params)

        # Select good points
        if current_points is not None and status is not None:
            good_new = current_points[status == 1]
            good_old = self.prev_points[status == 1]
        else:
            # print("Warning: Optical flow failed.")
            self.prev_gray = current_frame_gray.copy()
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            return None

        if len(good_new) < 4 : # Need at least 3 points for affine, 4 for robustness
            # print(f"Warning: Not enough good tracking points found ({len(good_new)}).")
            self.prev_gray = current_frame_gray.copy()
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            return None

        # Estimate rigid transform (or affine)
        # Using estimateRigidTransform is deprecated. Use estimateAffinePartial2D or estimateAffine2D
        # transform_matrix, _ = cv2.estimateAffinePartial2D(good_old, good_new)
        transform_matrix, _ = cv2.estimateAffine2D(good_old, good_new)


        if transform_matrix is None:
            # print("Warning: Could not estimate affine transform.")
            # Fallback or re-initialize points for next frame
            self.prev_gray = current_frame_gray.copy()
            self.prev_points = cv2.goodFeaturesToTrack(self.prev_gray, mask=None, **self.feature_params)
            return None # Could return identity or previous transform if desired

        # Update for next frame
        self.prev_gray = current_frame_gray.copy()
        self.prev_points = good_new.reshape(-1, 1, 2) # Reshape for next goodFeaturesToTrack input

        return transform_matrix

    def reset(self):
        """Resets the estimator for a new video sequence."""
        self.prev_gray = None
        self.prev_points = None

if __name__ == '__main__':
    # Example Usage: This requires a video sequence.
    # For a simple test, create two slightly translated dummy frames.
    frame1 = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(frame1, (100, 100), 30, 255, -1)
    cv2.putText(frame1, "F1", (150,150), cv2.FONT_HERSHEY_SIMPLEX,1, (200), 2)


    frame2 = np.zeros((240, 320), dtype=np.uint8)
    cv2.circle(frame2, (105, 102), 30, 255, -1) # Slightly moved
    cv2.putText(frame2, "F2", (155,152), cv2.FONT_HERSHEY_SIMPLEX,1, (200), 2)


    estimator = MotionEstimator(max_corners=50, quality_level=0.1, min_distance=7)

    # First frame processing
    print("Processing frame 1...")
    transform1 = estimator.estimate_transform(frame1)
    if transform1 is not None:
        print("Transform for frame 1 (relative to hypothetical prev=None, should be Identity or None):")
        print(transform1)
    else:
        print("No transform for frame 1 (as expected if it's the very first).")


    # Second frame processing
    print("\nProcessing frame 2...")
    transform2 = estimator.estimate_transform(frame2)
    if transform2 is not None:
        print("Transform from frame 1 to frame 2:")
        print(transform2)
        # Expected dx, dy should be around [5, 2]
        # Affine matrix is [[cos(th) sx, -sin(th) sy, tx], [sin(th) sx, cos(th) sy, ty]]
        # For pure translation: [[1, 0, tx], [0, 1, ty]]
        print(f"Estimated Translation: dx={transform2[0,2]:.2f}, dy={transform2[1,2]:.2f}")
    else:
        print("Could not estimate transform for frame 2.")

    # To run on a video:
    # cap = cv2.VideoCapture("your_video.mp4")
    # estimator = MotionEstimator()
    # while True:
    #     ret, frame = cap.read()
    #     if not ret: break
    #     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #     transform = estimator.estimate_transform(gray)
    #     if transform is not None:
    #         # `transform` is the motion from prev_gray to gray
    #         # Use this transform in motion_compensation
    #         pass
    #     cv2.imshow("Frame", frame)
    #     if cv2.waitKey(30) & 0xFF == ord('q'): break
    # cap.release()
    # cv2.destroyAllWindows()