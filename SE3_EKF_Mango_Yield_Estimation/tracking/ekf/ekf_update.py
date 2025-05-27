import numpy as np

class EKFUpdate:
    def __init__(self, measurement_model):
        """
        Handles the update step of the Extended Kalman Filter.
        Paper Appendix VII, Eq. 10-12 (measurement pred, residual),
        Eq. 15 (Innovation Cov S), Eq. 17 (Kalman Gain K),
        Eq. 18 (State Update), Eq. 19 (Covariance Update).

        Args:
            measurement_model (MeasurementModel): Instance for h(x) and Hk.
        """
        self.mm = measurement_model

    def update(self, predicted_state_xk_minus, predicted_covariance_Pk_minus,
               measurement_zk, measurement_noise_Rk):
        """
        Performs the EKF update step.

        Args:
            predicted_state_xk_minus (np.array): Predicted state x_k^{-}.
            predicted_covariance_Pk_minus (np.array): Predicted error covariance P_k^{-}.
            measurement_zk (np.array): Actual measurement z_k.
                                       [Cx, Cy, width, height, theta_obs]
            measurement_noise_Rk (np.array): Measurement noise covariance R_k.

        Returns:
            tuple:
                - updated_state_estimate_x_hat (np.array): Updated state estimate x_hat_{k|k}.
                - updated_covariance_P (np.array): Updated error covariance P_{k|k}.
        """
        # Measurement Prediction (Eq. 10 for h(x_k^-), Eq. 11 uses Hk which is Jacobian)
        # z_k_hat = h(x_k^{-})
        z_k_hat = self.mm.h(predicted_state_xk_minus)

        # Get Jacobian Hk (linearized measurement matrix)
        # Hk = dh/dx | at x_k^{-}
        Hk = self.mm.get_jacobian_Hk(predicted_state_xk_minus) # Pass current predicted state

        # Measurement Residual (Innovation) (Eq. 12)
        # y_k = z_k - z_k_hat
        # Handle angular wrap-around for theta residual if theta is part of measurement
        # theta_obs is measurement_zk[self.mm.THETA_OBS]
        # theta_pred is z_k_hat[self.mm.THETA_OBS]
        residual_yk = measurement_zk - z_k_hat
        
        # Angular residual: normalize to [-pi, pi]
        # The paper's state and measurement include theta directly.
        # If theta is an angle, its residual should be handled carefully (e.g., atan2(sin(diff), cos(diff)))
        # For simplicity, if theta is just a scalar value (e.g. from ORB keyfeatures as paper might imply for angle determination)
        # direct subtraction might be what's intended.
        # However, for true rotational angle, wrap-around is important.
        theta_obs_idx_in_zk = self.mm.THETA_OBS
        if theta_obs_idx_in_zk < len(residual_yk) : # If theta is indeed in measurement vector
            angle_diff = residual_yk[theta_obs_idx_in_zk]
            # Normalize angle_diff to [-pi, pi]
            angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi
            residual_yk[theta_obs_idx_in_zk] = angle_diff


        # Innovation Covariance (Eq. 15)
        # S_k = H_k * P_k^{-} * H_k^T + R_k
        Sk = Hk @ predicted_covariance_Pk_minus @ Hk.T + measurement_noise_Rk

        # Kalman Gain (Eq. 17)
        # K_k = P_k^{-} * H_k^T * S_k^{-1}
        try:
            Sk_inv = np.linalg.inv(Sk)
        except np.linalg.LinAlgError:
            # print("Warning: Innovation covariance Sk is singular. Using pseudo-inverse.")
            Sk_inv = np.linalg.pinv(Sk) # Use pseudo-inverse if Sk is singular

        Kk = predicted_covariance_Pk_minus @ Hk.T @ Sk_inv

        # Updated State Estimate (Eq. 18)
        # x_hat_{k|k} = x_k^{-} + K_k * y_k
        updated_state_estimate_x_hat = predicted_state_xk_minus + Kk @ residual_yk

        # Updated Covariance Estimate (Eq. 19)
        # P_{k|k} = (I - K_k * H_k) * P_k^{-}
        # More numerically stable form: Joseph form, or P = P_minus - K * S * K_T
        # Paper uses P = (I - KH)P_minus
        I = np.eye(predicted_state_xk_minus.shape[0], dtype=np.float32)
        updated_covariance_P = (I - Kk @ Hk) @ predicted_covariance_Pk_minus
        
        # Ensure P remains symmetric (due to potential numerical inaccuracies)
        updated_covariance_P = (updated_covariance_P + updated_covariance_P.T) / 2.0

        return updated_state_estimate_x_hat, updated_covariance_P

if __name__ == '__main__':
    from .state_vector import StateVector
    from .measurement_model import MeasurementModel
    import yaml

    # Dummy predicted state and covariance (output from EKFPredict)
    # [x1, y1, x2, y2, vx1, vy1, vx2, vy2, theta, delta_theta]
    x_k_minus_example = np.array([11, 20.5, 111.5, 120.8, 1, 0.5, 1.5, 0.8, 0.12, 0.02], dtype=np.float32)
    P_k_minus_diag_values = [15.0] * StateVector.DIM # Example, slightly larger than P_prev
    P_k_minus_example = np.diag(P_k_minus_diag_values).astype(np.float32)

    # Actual measurement zk = [Cx, Cy, width, height, theta_obs]
    # Suppose observed mango is at Cx=62, Cy=71, W=98, H=101, Theta_obs=0.11
    zk_actual = np.array([62.0, 71.0, 98.0, 101.0, 0.11], dtype=np.float32)

    # Load example config for Rk (measurement noise)
    # This would typically come from configs/tracking.yaml
    dummy_config_r_diag = {
        'center_x_r': 10.0, 'center_y_r': 10.0,
        'width_r': 10.0, 'height_r': 10.0, 'obs_theta_r': 0.01
    }
    R_diag_values = [
        dummy_config_r_diag['center_x_r'], dummy_config_r_diag['center_y_r'],
        dummy_config_r_diag['width_r'], dummy_config_r_diag['height_r'],
        dummy_config_r_diag['obs_theta_r']
    ]
    Rk_example = np.diag(R_diag_values).astype(np.float32)


    mm_instance = MeasurementModel()
    ekf_updater = EKFUpdate(measurement_model=mm_instance)

    print(f"Predicted state (x_k_minus):\n{x_k_minus_example}")
    print(f"\nPredicted covariance (P_k_minus) diag:\n{np.diag(P_k_minus_example)}")
    print(f"\nActual measurement (zk):\n{zk_actual}")
    print(f"\nMeasurement noise (Rk) diag:\n{np.diag(Rk_example)}")

    updated_x_hat, updated_P = ekf_updater.update(x_k_minus_example, P_k_minus_example,
                                                  zk_actual, Rk_example)

    print(f"\nUpdated state estimate (x_hat_k_k):\n{updated_x_hat}")
    print(f"\nUpdated covariance (P_k_k) diag:\n{np.diag(updated_P)}")

    # Check if state moved towards measurement
    # Expected measurement from x_k_minus_example:
    # Cx=(11+111.5)/2=61.25, Cy=(20.5+120.8)/2=70.65, W=100.5, H=100.3, Th=0.12
    # zk_actual = [62, 71, 98, 101, 0.11]
    # The updated state should reflect a shift towards zk_actual values.
    # e.g. updated_x_hat[StateVector.THETA] should be between 0.12 and 0.11
    
    # Check if covariance decreased (diagonal elements)
    assert np.all(np.diag(updated_P) < np.diag(P_k_minus_example) + 1e-5), "Covariance did not decrease after update"
    assert updated_P.shape == (StateVector.DIM, StateVector.DIM)