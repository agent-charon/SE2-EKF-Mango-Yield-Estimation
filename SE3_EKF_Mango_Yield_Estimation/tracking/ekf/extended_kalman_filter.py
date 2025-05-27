import numpy as np
import yaml
from .state_vector import StateVector
from .state_transition import StateTransitionModel
from .measurement_model import MeasurementModel
from .ekf_predict import EKFPredict
from .ekf_update import EKFUpdate

class ExtendedKalmanFilter:
    def __init__(self, track_id, initial_bbox_xyxy, initial_theta, config_path="configs/tracking.yaml"):
        """
        Initializes a single Extended Kalman Filter tracker instance.

        Args:
            track_id (int): Unique ID for this track.
            initial_bbox_xyxy (list or np.array): [x1, y1, x2, y2] for the first detection.
            initial_theta (float): Initial observed rotational angle for the bbox.
            config_path (str): Path to the tracking configuration YAML file.
        """
        self.track_id = track_id
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)['ekf']
        except FileNotFoundError:
            print(f"Error: Tracking config file not found at {config_path}. Using default EKF parameters.")
            # Provide some fallback defaults if config is missing - not ideal for production
            config = {
                'delta_t': 1.0,
                'initial_covariance_diag': {k: 10.0 for k in ['pos_tl_x', 'pos_tl_y', 'pos_br_x', 'pos_br_y', 'vel_tl_x', 'vel_tl_y', 'vel_br_x', 'vel_br_y', 'angle_theta', 'angle_delta_theta']},
                'process_noise_diag': {k: 0.1 for k in ['q_x1', 'q_y1', 'q_x2', 'q_y2', 'q_vx1', 'q_vy1', 'q_vx2', 'q_vy2', 'q_theta', 'q_delta_theta']},
                'measurement_noise_diag': {k: 1.0 for k in ['center_x_r', 'center_y_r', 'width_r', 'height_r', 'obs_theta_r']}
            }

        self.dt = float(config.get('delta_t', 1.0))

        # Initialize EKF components
        self.stm = StateTransitionModel(dt=self.dt)
        self.mm = MeasurementModel()
        self.predictor = EKFPredict(state_transition_model=self.stm)
        self.updater = EKFUpdate(measurement_model=self.mm)

        # Initialize state and covariance
        self.x_hat = StateVector.bbox_to_state(initial_bbox_xyxy, initial_theta=initial_theta) # x_hat_{k-1|k-1}
        
        # Initial Covariance P0
        p_diag_conf = config.get('initial_covariance_diag', {})
        P0_diag = np.array([
            p_diag_conf.get('pos_tl_x', 100.0), p_diag_conf.get('pos_tl_y', 100.0),
            p_diag_conf.get('pos_br_x', 100.0), p_diag_conf.get('pos_br_y', 100.0),
            p_diag_conf.get('vel_tl_x', 25.0), p_diag_conf.get('vel_tl_y', 25.0),
            p_diag_conf.get('vel_br_x', 25.0), p_diag_conf.get('vel_br_y', 25.0),
            p_diag_conf.get('angle_theta', 0.1), p_diag_conf.get('angle_delta_theta', 0.01)
        ], dtype=np.float32)
        self.P = np.diag(P0_diag) # P_{k-1|k-1}

        # Process Noise Qk
        q_diag_conf = config.get('process_noise_diag', {})
        Q_diag = np.array([
            q_diag_conf.get('q_x1', 1.0), q_diag_conf.get('q_y1', 1.0),
            q_diag_conf.get('q_x2', 1.0), q_diag_conf.get('q_y2', 1.0),
            q_diag_conf.get('q_vx1', 0.25), q_diag_conf.get('q_vy1', 0.25),
            q_diag_conf.get('q_vx2', 0.25), q_diag_conf.get('q_vy2', 0.25),
            q_diag_conf.get('q_theta', 0.01), q_diag_conf.get('q_delta_theta', 0.001)
        ], dtype=np.float32)
        self.Qk = np.diag(Q_diag)

        # Measurement Noise Rk
        r_diag_conf = config.get('measurement_noise_diag', {})
        R_diag = np.array([
            r_diag_conf.get('center_x_r', 10.0), r_diag_conf.get('center_y_r', 10.0),
            r_diag_conf.get('width_r', 10.0), r_diag_conf.get('height_r', 10.0),
            r_diag_conf.get('obs_theta_r', 0.01)
        ], dtype=np.float32)
        self.Rk = np.diag(R_diag)

        # Track attributes
        self.time_since_update = 0
        self.hits = 1 # Number of times this track has been successfully updated
        self.age = 0 # Age of the track in terms of filter updates

    def predict(self):
        """Performs the prediction step for this track."""
        self.x_hat_minus, self.P_minus = self.predictor.predict(self.x_hat, self.P, self.Qk)
        self.age += 1
        self.time_since_update += 1
        return self.x_hat_minus # Return predicted state (e.g., for matching)

    def update(self, measurement_bbox_cxcywh_theta):
        """
        Performs the update step for this track with a new measurement.
        Args:
            measurement_bbox_cxcywh_theta (np.array): Measurement [Cx, Cy, W, H, Theta_obs].
        """
        if len(measurement_bbox_cxcywh_theta) != self.mm.MEAS_DIM:
            raise ValueError(f"Measurement vector has wrong dimension. Expected {self.mm.MEAS_DIM}, got {len(measurement_bbox_cxcywh_theta)}")
        
        # x_hat_minus and P_minus should have been set by a preceding predict() call
        if not hasattr(self, 'x_hat_minus') or not hasattr(self, 'P_minus'):
            # This case should ideally not happen if predict() is always called first.
            # If it's the very first update after initialization, predict might not have run.
            # For robustness, run predict if x_hat_minus is not available.
            # print("Warning: EKF update called without prior predict. Predicting now.")
            self.predict() # This advances age and time_since_update

        self.x_hat, self.P = self.updater.update(self.x_hat_minus, self.P_minus,
                                                 measurement_bbox_cxcywh_theta, self.Rk)
        self.time_since_update = 0
        self.hits += 1
    
    def get_current_state_bbox_xyxy(self):
        """Returns the current estimated bounding box [x1, y1, x2, y2]."""
        return StateVector.state_to_bbox_xyxy(self.x_hat)

    def get_current_state_cxcywh_theta(self):
        """Returns current estimated [Cx, Cy, W, H, Theta]."""
        return StateVector.state_to_bbox_cxcywh_theta(self.x_hat)

    def get_predicted_bbox_xyxy(self):
        """Returns the predicted bounding box [x1, y1, x2, y2] from x_hat_minus."""
        if hasattr(self, 'x_hat_minus'):
            return StateVector.state_to_bbox_xyxy(self.x_hat_minus)
        return StateVector.state_to_bbox_xyxy(self.x_hat) # Fallback to current estimate if no prediction yet


if __name__ == '__main__':
    # Example usage:
    initial_mango_bbox = [50, 50, 150, 150] # x1,y1,x2,y2
    initial_mango_angle = 0.0 # radians

    # Create a dummy tracking.yaml if not present for testing
    dummy_config_content = """
ekf:
  delta_t: 1.0
  initial_covariance_diag:
    pos_tl_x: 100.0; pos_tl_y: 100.0; pos_br_x: 100.0; pos_br_y: 100.0
    vel_tl_x: 25.0; vel_tl_y: 25.0; vel_br_x: 25.0; vel_br_y: 25.0
    angle_theta: 0.1; angle_delta_theta: 0.01
  process_noise_diag:
    q_x1: 1.0; q_y1: 1.0; q_x2: 1.0; q_y2: 1.0
    q_vx1: 0.25; q_vy1: 0.25; q_vx2: 0.25; q_vy2: 0.25
    q_theta: 0.01; q_delta_theta: 0.001
  measurement_noise_diag:
    center_x_r: 10.0; center_y_r: 10.0
    width_r: 10.0; height_r: 10.0; obs_theta_r: 0.01
"""
    # This dummy config uses semicolons, YAML uses colons. Correcting.
    dummy_config_content_yaml = """
ekf:
  delta_t: 1.0
  initial_covariance_diag:
    pos_tl_x: 100.0
    pos_tl_y: 100.0
    pos_br_x: 100.0
    pos_br_y: 100.0
    vel_tl_x: 25.0
    vel_tl_y: 25.0
    vel_br_x: 25.0
    vel_br_y: 25.0
    angle_theta: 0.1
    angle_delta_theta: 0.01
  process_noise_diag:
    q_x1: 1.0
    q_y1: 1.0
    q_x2: 1.0
    q_y2: 1.0
    q_vx1: 0.25
    q_vy1: 0.25
    q_vx2: 0.25
    q_vy2: 0.25
    q_theta: 0.01
    q_delta_theta: 0.001
  measurement_noise_diag:
    center_x_r: 10.0
    center_y_r: 10.0
    width_r: 10.0
    height_r: 10.0
    obs_theta_r: 0.01
"""
    test_config_path = "temp_test_tracking_config.yaml"
    with open(test_config_path, "w") as f:
        f.write(dummy_config_content_yaml)


    mango_tracker = ExtendedKalmanFilter(track_id=1,
                                         initial_bbox_xyxy=initial_mango_bbox,
                                         initial_theta=initial_mango_angle,
                                         config_path=test_config_path)

    print(f"Initialized EKF for track ID {mango_tracker.track_id}")
    print(f"Initial state (x_hat):\n{mango_tracker.x_hat}")
    print(f"Initial P diag:\n{np.diag(mango_tracker.P)}")

    # Simulate a prediction step
    predicted_state = mango_tracker.predict()
    print(f"\nAfter 1st prediction step:")
    print(f"  Predicted state (x_hat_minus):\n{predicted_state}")
    print(f"  Age: {mango_tracker.age}, Hits: {mango_tracker.hits}, Time Since Update: {mango_tracker.time_since_update}")

    # Simulate a measurement
    # Cx, Cy, W, H, Theta_obs
    # Initial bbox was [50,50,150,150] -> Cx=100, Cy=100, W=100, H=100, Th=0
    # Assume it moved slightly and rotated
    measurement1 = np.array([102.0, 101.0, 100.0, 100.0, 0.05], dtype=np.float32) 
    mango_tracker.update(measurement1)
    print(f"\nAfter 1st update with measurement: {measurement1}")
    print(f"  Updated state (x_hat):\n{mango_tracker.x_hat}")
    print(f"  Updated P diag:\n{np.diag(mango_tracker.P)}")
    print(f"  Age: {mango_tracker.age}, Hits: {mango_tracker.hits}, Time Since Update: {mango_tracker.time_since_update}")
    
    current_box = mango_tracker.get_current_state_bbox_xyxy()
    print(f"  Current estimated bbox [x1,y1,x2,y2]: {current_box}")

    # Clean up dummy config
    import os
    os.remove(test_config_path)