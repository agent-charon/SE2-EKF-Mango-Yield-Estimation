import cv2
import numpy as np

class MotionCompensator:
    def __init__(self):
        """
        Applies motion compensation to a frame using a smoothed transformation.
        """
        pass # No state needed for this class, it's a stateless operation per frame

    def compensate_frame(self, frame, smoothed_affine_transform):
        """
        Warps the current frame to compensate for camera motion.
        The `smoothed_affine_transform` should be the transform that maps
        the *current* frame's view to the *stabilized* view (or to align with the previous stabilized frame).

        The logic for stabilization is:
        1. Estimate T_k: transform from frame k-1 to frame k.
        2. Smooth T_k sequence to get T_smooth_k.
        3. The compensation transform for frame k is to undo (T_k - T_smooth_k).
           Or, more commonly, accumulate T_k to get trajectory P_k. Smooth P_k to get P_smooth_k.
           The transform to apply to frame k is P_smooth_k * inv(P_k).
           This is complex.

        A simpler interpretation from the paper's "three-stage video stabilization process:
        motion estimation, motion smoothing, and motion compensation":
        - Motion Estimation gives T_k (transform from frame k-1 to k)
        - Motion Smoothing gives T_smooth_k (smoothed version of T_k)
        - Motion Compensation: To stabilize frame_k, we want it to look like it underwent T_smooth_k
          instead of T_k relative to frame_k-1.
          The transform to apply to frame_k is effectively one that counters the "jitter" (T_k - T_smooth_k).
          Warp frame_k with inv(T_k) * T_smooth_k.
          Or, if T_smooth_k is the desired motion from a fixed reference,
          and we have a cumulative transform C_k to frame_k, and desired cumulative C_smooth_k,
          then warp frame_k with C_smooth_k * inv(C_k).

        Let's assume `smoothed_affine_transform` is the desired transform
        that frame_k should undergo relative to some stable reference.
        Or, it's the transform to map current_frame to the stabilized coordinate system.

        If `smoothed_affine_transform` is the (dx, dy, da) parameters:
        dx_s, dy_s, da_s are the smoothed motion from frame k-1 to frame k.
        We need to build an affine matrix from this.
        M_smooth_k = [[cos(da_s), -sin(da_s), dx_s], [sin(da_s), cos(da_s), dy_s]]
        The frame `frame` is frame_k. We warp it with M_smooth_k.
        This assumes we are warping relative to a perfectly stable previous frame.

        Let's reconsider. Typically, video stabilization accumulates transforms.
        - C_k = T_k * T_{k-1} * ... * T_1 (Cumulative transform to frame k)
        - Smooth this C_k trajectory to get C_smooth_k.
        - The transform to apply to current frame_k is C_smooth_{k-1} * inv(C_{k-1}).
          No, this is for stabilizing frame k to align with frame k-1's smoothed position.
          The transform to apply to frame_k to get stabilized_frame_k is: M_stabilize = C_smooth_k * inv(C_k).
          Then: stabilized_frame_k = warp(frame_k, M_stabilize).

        The MotionSmoother provided smoothed *inter-frame* motions (dx_s, dy_s, da_s).
        Let (dx_raw, dy_raw, da_raw) be the raw inter-frame motion for current frame.
        The correction transform applied to the current frame is one that makes its motion
        become (dx_s, dy_s, da_s) instead of (dx_raw, dy_raw, da_raw).
        If M_raw is the affine from (dx_r, dy_r, da_r)
        If M_smooth is the affine from (dx_s, dy_s, da_s)
        We apply M_compensate = M_smooth * inv(M_raw) to the current frame.

        Args:
            frame (np.array): The current frame to be stabilized.
            raw_motion_params (tuple): (dx_raw, dy_raw, da_raw) estimated for this frame relative to prev.
            smoothed_motion_params (tuple): (dx_s, dy_s, da_s) target motion for this frame relative to prev.

        Returns:
            np.array: The motion-compensated (stabilized) frame.
        """
        if frame is None:
            print("Warning: MotionCompensator received a None frame.")
            return None
        
        # This function will take the current frame, the original estimated transform (T_k)
        # and the smoothed transform (T_smooth_k).
        # The goal is to warp frame_k so that its motion from frame_{k-1} becomes T_smooth_k.
        # The original motion was T_k.
        # Warp_Transform = T_smooth_k * inv(T_k)
        # stabilized_frame_k = cv2.warpAffine(frame_k, Warp_Transform, (w,h))

        # This will be orchestrated by a main stabilization loop.
        # This `compensate_frame` function just does the warping.
        # `smoothed_affine_transform` here is the final transform to apply to the frame.
        
        if smoothed_affine_transform is None:
            return frame # No compensation if transform is None

        rows, cols = frame.shape[:2]
        stabilized_frame = cv2.warpAffine(frame, smoothed_affine_transform, (cols, rows),
                                          borderMode=cv2.BORDER_REPLICATE) # or BORDER_CONSTANT
        return stabilized_frame

if __name__ == '__main__':
    compensator = MotionCompensator()

    # Dummy frame
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(frame, "Original", (50, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Suppose this frame underwent a raw transform T_k (e.g., shift by (10,5))
    # T_k = [[1, 0, 10], [0, 1, 5]]
    # Suppose the desired smoothed transform T_smooth_k is (shift by (2,1))
    # T_smooth_k = [[1, 0, 2], [0, 1, 1]]

    # The compensation matrix M_comp = T_smooth_k * inv(T_k)
    # inv(T_k) = [[1, 0, -10], [0, 1, -5]] (for pure translation)
    # M_comp = [[1,0,2],[0,1,1]] @ [[1,0,-10],[0,1,-5],[0,0,1]] (using 3x3 for mult)
    # M_comp = [[1,0,2],[0,1,1]] @ [[1,0,-10],[0,1,-5]] (if T_k is 2x3, its inverse in this context is tricky)
    
    # Let's use the definition that `smoothed_affine_transform` IS the M_comp to apply.
    # This means the caller calculates M_smooth_k * inv(T_k).
    # For example, to correct a +10px horizontal shake to +0px:
    # The transform to apply would be a shift of -10px.
    M_example_compensation = np.array([[1.0, 0.0, -10.0],  # Shift left by 10
                                       [0.0, 1.0, 0.0]], dtype=np.float32)

    stabilized_frame = compensator.compensate_frame(frame.copy(), M_example_compensation)

    print("Applied example compensation.")

    # Display (optional)
    # cv2.imshow("Original Frame", frame)
    # cv2.imshow("Stabilized Frame (Example)", stabilized_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # A more realistic stabilization loop would be in a main script:
    # prev_frame_gray = None
    # current_frame_gray = ...
    # T_k = motion_estimator.estimate_transform(current_frame_gray, prev_frame_gray)
    # motion_smoother.add_transform_params(dx_k, dy_k, da_k) # from T_k
    # all_smoothed_params = motion_smoother.get_smoothed_params_for_all_frames()
    # smoothed_dx_curr, smoothed_dy_curr, smoothed_da_curr = all_smoothed_params[current_frame_index]
    
    # M_raw_k = build_affine(dx_k, dy_k, da_k)
    # M_smooth_k = build_affine(smoothed_dx_curr, smoothed_dy_curr, smoothed_da_curr)
    
    # M_compensation = M_smooth_k @ np.linalg.inv(np.vstack([M_raw_k, [0,0,1]]))[:2,:] # If M_raw_k is invertible
    # stabilized_frame = compensator.compensate_frame(current_frame, M_compensation)