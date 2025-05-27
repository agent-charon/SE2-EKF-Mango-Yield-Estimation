import numpy as np

class MotionSmoother:
    def __init__(self, smoothing_radius=15):
        """
        Smooths a series of transformation parameters using a moving average filter.
        The paper mentions a "moving average filter". This typically applies to the
        translation, scale, and rotation components derived from the affine matrices.

        Args:
            smoothing_radius (int): The radius of the moving average window.
                                    Window size will be 2 * radius + 1.
                                    The paper implies this is applied to remove "unwanted movements".
        """
        if smoothing_radius <= 0:
            raise ValueError("Smoothing radius must be positive.")
        self.smoothing_radius = smoothing_radius
        self.transform_history = [] # Stores (dx, dy, da) where da is change in angle

    def _decompose_affine(self, affine_matrix):
        """
        Decomposes a 2x3 affine matrix into translation, rotation angle, and scale.
        Note: Full decomposition of affine into scale_x, scale_y, shear, rotation, translation
        is complex if scales are non-uniform or shear is present.
        For stabilization, often a simpler model (translation + rotation + uniform scale) is assumed.
        cv2.estimateAffinePartial2D returns a similarity transform (translation, rotation, uniform scale).
        cv2.estimateAffine2D can return a full affine.

        If using estimateAffine2D, true decomposition is:
        A = T * R * S * Sh (Translation, Rotation, Scale, Shear)
        For this, we'll simplify and assume it's mostly translation, rotation, and perhaps uniform scale.
        dx = affine_matrix[0, 2]
        dy = affine_matrix[1, 2]
        angle_rad = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
        scale = np.sqrt(affine_matrix[0,0]**2 + affine_matrix[1,0]**2) # Assuming uniform scale
        return dx, dy, angle_rad, scale
        """
        dx = affine_matrix[0, 2]
        dy = affine_matrix[1, 2]
        # Rotation angle
        # a = M[0,0], b = M[1,0] -> angle = atan2(b,a)
        da = np.arctan2(affine_matrix[1, 0], affine_matrix[0, 0])
        # Scale (assuming uniform scale from the rotational part)
        # sx = M[0,0]/cos(da), sy = M[1,1]/cos(da)
        # For simplicity, let's just track dx, dy, da for smoothing
        # scale_x = np.sqrt(affine_matrix[0,0]**2 + affine_matrix[1,0]**2) # This is not scale_x if there's rotation
        # scale_y = np.sqrt(affine_matrix[0,1]**2 + affine_matrix[1,1]**2) # This is not scale_y
        return dx, dy, da


    def add_transform(self, affine_matrix):
        """
        Adds a new affine transformation matrix to the history for smoothing.
        The matrix represents the transform from frame k-1 to frame k.

        Args:
            affine_matrix (np.array): 2x3 affine transformation matrix.
        """
        if affine_matrix is None:
            # If no transform, assume no motion relative to last known good transform
            # For smoothing, we need consistent data.
            # If this happens often, stabilization quality will degrade.
            # A simple approach: use the last valid motion parameters.
            if self.transform_history:
                self.transform_history.append(self.transform_history[-1])
            else: # No history yet, assume zero motion (dx,dy,da)
                self.transform_history.append((0.0, 0.0, 0.0))
            return

        dx, dy, da = self._decompose_affine(affine_matrix)
        self.transform_history.append((dx, dy, da))

    def get_smoothed_trajectory(self):
        """
        Calculates the smoothed trajectory of transformations.
        This method computes cumulative transforms and then smooths them.

        Returns:
            list of tuples: List of (smoothed_dx, smoothed_dy, smoothed_da)
                            representing the smoothed *motion* for each frame relative to the previous.
        """
        if not self.transform_history:
            return []

        # 1. Accumulate transformations to get a trajectory (x, y, angle)
        # This is the position/orientation of the camera if frame 0 is origin
        trajectory_x = [0.0]
        trajectory_y = [0.0]
        trajectory_angle = [0.0]

        for i in range(len(self.transform_history)):
            dx, dy, da = self.transform_history[i] # Motion from frame i-1 to i
            
            # This accumulation logic assumes dx,dy are in a consistent global frame
            # or that da correctly rotates them.
            # If dx, dy, da are from M_k = M_{k-1}_to_k * M_{k-1}, this is complex.
            # Simpler: The transform_history stores INTER-FRAME motion (dx_k, dy_k, da_k).
            # We want to smooth this sequence directly.
            prev_x = trajectory_x[-1]
            prev_y = trajectory_y[-1]
            prev_angle = trajectory_angle[-1]
            
            # Update position based on previous angle and current inter-frame motion
            # This step is tricky. The dx, dy from estimateAffine are in the *previous* frame's coord system.
            # For a global trajectory:
            # x_k = x_{k-1} + dx_k * cos(angle_{k-1}) - dy_k * sin(angle_{k-1})
            # y_k = y_{k-1} + dx_k * sin(angle_{k-1}) + dy_k * cos(angle_{k-1})
            # angle_k = angle_{k-1} + da_k
            # However, the paper's "moving average filter to remove unwanted movements" likely means
            # smoothing the *inter-frame* dx, dy, da values themselves, not the cumulative path.
            pass # We will smooth the dx, dy, da sequence directly.


        # Extract sequences of dx, dy, da
        dx_sequence = np.array([t[0] for t in self.transform_history])
        dy_sequence = np.array([t[1] for t in self.transform_history])
        da_sequence = np.array([t[2] for t in self.transform_history])

        # 2. Smooth these sequences using a moving average
        smoothed_dx_sequence = self._moving_average(dx_sequence)
        smoothed_dy_sequence = self._moving_average(dy_sequence)
        smoothed_da_sequence = self._moving_average(da_sequence)
        
        # The result should be a list of smoothed (dx, dy, da) for each frame transition
        smoothed_motion_params = []
        for i in range(len(self.transform_history)):
            smoothed_motion_params.append(
                (smoothed_dx_sequence[i], smoothed_dy_sequence[i], smoothed_da_sequence[i])
            )
        return smoothed_motion_params


    def _moving_average(self, curve):
        """Applies a moving average filter to a 1D curve."""
        window_size = 2 * self.smoothing_radius + 1
        if len(curve) < window_size: # If curve is too short, no smoothing or less effective
            return curve # Or handle with padding/edge cases if necessary
            
        # Pad the curve at the beginning and end to handle edges
        # 'reflect' padding is often good for signals
        padded_curve = np.pad(curve, (self.smoothing_radius, self.smoothing_radius), 'edge') # or 'reflect'
        
        # Use a convolution for moving average
        kernel = np.ones(window_size) / window_size
        smoothed_curve = np.convolve(padded_curve, kernel, mode='valid') # 'valid' ensures output is same length as original curve
        
        if len(smoothed_curve) != len(curve): # Should match if padding and mode='valid' are correct
            # Fallback if convolution gives unexpected length (e.g. if curve was shorter than window)
            # This might happen if curve len < window_size initially
             if len(curve) < window_size :
                # Create a simple moving average for short curves
                smoothed_curve_manual = []
                for i in range(len(curve)):
                    start = max(0, i - self.smoothing_radius)
                    end = min(len(curve), i + self.smoothing_radius + 1)
                    smoothed_curve_manual.append(np.mean(curve[start:end]))
                return np.array(smoothed_curve_manual)
             else: # Should not happen with correct padding
                print(f"Warning: Smoothed curve length mismatch. Original: {len(curve)}, Smoothed: {len(smoothed_curve)}")
                return curve # Return original if smoothing fails unexpectedly


        return smoothed_curve

    def reset(self):
        self.transform_history = []


if __name__ == '__main__':
    # Example usage
    smoother = MotionSmoother(smoothing_radius=2) # Small radius for example

    # Simulate some jerky inter-frame motion parameters (dx, dy, da)
    # (transform from frame k-1 to k)
    raw_motions = [
        (1.0, 0.5, 0.01),  # Frame 0 -> 1
        (5.0, 4.5, 0.05),  # Frame 1 -> 2 (jerky)
        (1.2, 0.6, 0.012), # Frame 2 -> 3
        (0.8, 0.3, 0.008), # Frame 3 -> 4
        (6.0, 5.5, 0.06),  # Frame 4 -> 5 (jerky)
        (1.1, 0.4, 0.009), # Frame 5 -> 6
    ]

    print("Raw inter-frame motions (dx, dy, da):")
    for i, motion in enumerate(raw_motions):
        # Simulate affine matrix (only dx, dy, da components matter for this smoother)
        # M = [[cos(da), -sin(da), dx], [sin(da), cos(da), dy]]
        da = motion[2]
        dummy_affine = np.array([
            [np.cos(da), -np.sin(da), motion[0]],
            [np.sin(da),  np.cos(da), motion[1]]
        ], dtype=np.float32)
        smoother.add_transform(dummy_affine)
        print(f"  Frame {i} to {i+1}: dx={motion[0]:.2f}, dy={motion[1]:.2f}, da={motion[2]:.3f}")


    smoothed_trajectory_params = smoother.get_smoothed_trajectory()

    print("\nSmoothed inter-frame motions (dx, dy, da):")
    for i, params in enumerate(smoothed_trajectory_params):
        print(f"  Frame {i} to {i+1}: dx={params[0]:.2f}, dy={params[1]:.2f}, da={params[2]:.3f}")
    
    # Test with None transforms
    smoother.reset()
    smoother.add_transform(np.array([[1,0,1],[0,1,1]], dtype=np.float32))
    smoother.add_transform(None) # Simulate a lost transform
    smoother.add_transform(np.array([[1,0,2],[0,1,2]], dtype=np.float32))
    smoothed_none_test = smoother.get_smoothed_trajectory()
    print("\nSmoothed with None transform in between:")
    for i, params in enumerate(smoothed_none_test):
        print(f"  Frame {i} to {i+1}: dx={params[0]:.2f}, dy={params[1]:.2f}, da={params[2]:.3f}")