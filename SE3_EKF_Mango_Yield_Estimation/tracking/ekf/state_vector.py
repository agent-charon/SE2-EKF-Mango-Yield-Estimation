import numpy as np

class StateVector:
    # State vector xk = [x1, y1, x2, y2, vx1, vy1, vx2, vy2, theta, delta_theta]
    # x1, y1: top-left corner
    # x2, y2: bottom-right corner
    # vx1, vy1: velocity of top-left corner
    # vx2, vy2: velocity of bottom-right corner
    # theta: rotational angle of the bounding box (e.g., radians)
    # delta_theta: rate of change of rotational angle (angular velocity)

    # Indices for state vector components
    X1, Y1, X2, Y2 = 0, 1, 2, 3
    VX1, VY1, VX2, VY2 = 4, 5, 6, 7
    THETA, DELTA_THETA = 8, 9
    DIM = 10 # Dimension of the state vector

    @staticmethod
    def bbox_to_state(bbox_xyxy, initial_theta=0.0):
        """
        Converts a bounding box [x1, y1, x2, y2] to an initial state vector.
        Velocities and delta_theta are initialized to zero.
        Args:
            bbox_xyxy (list or np.array): [x1, y1, x2, y2]
            initial_theta (float): Initial rotational angle.
        Returns:
            np.array: Initial state vector.
        """
        x1, y1, x2, y2 = bbox_xyxy
        return np.array([x1, y1, x2, y2, 0, 0, 0, 0, initial_theta, 0], dtype=np.float32)

    @staticmethod
    def state_to_bbox_xyxy(state_vector):
        """Extracts [x1, y1, x2, y2] from the state vector."""
        return state_vector[[StateVector.X1, StateVector.Y1, StateVector.X2, StateVector.Y2]].copy()
    
    @staticmethod
    def state_to_bbox_cxcywh_theta(state_vector):
        """Converts state vector to [center_x, center_y, width, height, theta]."""
        x1, y1, x2, y2 = StateVector.state_to_bbox_xyxy(state_vector)
        theta = state_vector[StateVector.THETA]
        
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return np.array([center_x, center_y, width, height, theta], dtype=np.float32)

    @staticmethod
    def get_rotational_angle(state_vector):
        """Extracts the rotational angle theta from the state vector."""
        return state_vector[StateVector.THETA]

    @staticmethod
    def get_angular_velocity(state_vector):
        """Extracts the angular velocity delta_theta from the state vector."""
        return state_vector[StateVector.DELTA_THETA]

if __name__ == '__main__':
    bbox = [10, 20, 110, 120] # x1, y1, x2, y2
    initial_angle_rad = np.pi / 4 # 45 degrees

    initial_state = StateVector.bbox_to_state(bbox, initial_theta=initial_angle_rad)
    print(f"Initial BBox: {bbox}")
    print(f"Initial State Vector (x1,y1,x2,y2, vx1,vy1,vx2,vy2, theta, dtheta):\n{initial_state}")

    extracted_bbox = StateVector.state_to_bbox_xyxy(initial_state)
    print(f"\nExtracted BBox from state: {extracted_bbox}")
    assert np.allclose(bbox, extracted_bbox)

    cxcywh_theta = StateVector.state_to_bbox_cxcywh_theta(initial_state)
    print(f"\nState to cx, cy, w, h, theta: {cxcywh_theta}")
    assert np.isclose(cxcywh_theta[0], (10+110)/2)
    assert np.isclose(cxcywh_theta[2], 100)
    assert np.isclose(cxcywh_theta[4], initial_angle_rad)

    angle = StateVector.get_rotational_angle(initial_state)
    print(f"\nExtracted Rotational Angle: {angle:.4f} radians")
    assert np.isclose(angle, initial_angle_rad)

    ang_vel = StateVector.get_angular_velocity(initial_state)
    print(f"Extracted Angular Velocity: {ang_vel:.4f} rad/update")
    assert np.isclose(ang_vel, 0.0)