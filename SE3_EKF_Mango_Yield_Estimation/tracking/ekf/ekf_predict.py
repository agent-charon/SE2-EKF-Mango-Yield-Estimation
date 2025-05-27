import numpy as np

class EKFPredict:
    def __init__(self, state_transition_model):
        """
        Handles the prediction step of the Extended Kalman Filter.
        Paper Appendix VII, Eq. 6 (state prediction), Eq. 8 (covariance prediction).
        x_k_minus = f(x_k_minus_1, u_k)  (Here u_k is assumed zero or part of Q)
                  => x_k_minus = Fk * x_k_minus_1 (for linear system, or f(x) for non-linear)
        P_k_minus = Fk * P_k_minus_1 * Fk_T + Qk

        Args:
            state_transition_model (StateTransitionModel): Instance for f(x) and Fk.
        """
        self.stm = state_transition_model

    def predict(self, prev_state_estimate_x_hat, prev_covariance_P, process_noise_Q):
        """
        Performs the EKF prediction step.

        Args:
            prev_state_estimate_x_hat (np.array): Previous state estimate x_hat_{k-1|k-1}.
            prev_covariance_P (np.array): Previous error covariance P_{k-1|k-1}.
            process_noise_Q (np.array): Process noise covariance Q_k.

        Returns:
            tuple:
                - predicted_state_xk_minus (np.array): Predicted state x_k^{-}.
                - predicted_covariance_Pk_minus (np.array): Predicted error covariance P_k^{-}.
        """
        # State Prediction (Eq. 6, using the non-linear function f)
        # x_k^{-} = f(x_hat_{k-1|k-1}, u_{k-1})
        # Here, u_{k-1} (control input) is assumed to be zero or incorporated into the model/Q.
        # The paper's Eq. 4 uses Fk * xk-1 + uk. If stm.predict_next_state is f(x), it's fine.
        predicted_state_xk_minus = self.stm.predict_next_state(prev_state_estimate_x_hat)

        # Get Jacobian Fk (linearized state transition matrix)
        # Fk = df/dx | at x_hat_{k-1|k-1}
        Fk = self.stm.get_jacobian_Fk(prev_state_estimate_x_hat) # Pass current state for linearization if needed

        # Covariance Prediction (Eq. 8)
        # P_k^{-} = F_k * P_{k-1|k-1} * F_k^T + Q_k
        predicted_covariance_Pk_minus = Fk @ prev_covariance_P @ Fk.T + process_noise_Q
        
        return predicted_state_xk_minus, predicted_covariance_Pk_minus

if __name__ == '__main__':
    from .state_vector import StateVector
    from .state_transition import StateTransitionModel
    import yaml

    # Load example config for Q (process noise)
    # This would typically come from configs/tracking.yaml
    dummy_config_q_diag = {
        'q_x1': 1.0, 'q_y1': 1.0, 'q_x2': 1.0, 'q_y2': 1.0,
        'q_vx1': 0.25, 'q_vy1': 0.25, 'q_vx2': 0.25, 'q_vy2': 0.25,
        'q_theta': 0.01, 'q_delta_theta': 0.001
    }
    Q_diag_values = [
        dummy_config_q_diag['q_x1'], dummy_config_q_diag['q_y1'],
        dummy_config_q_diag['q_x2'], dummy_config_q_diag['q_y2'],
        dummy_config_q_diag['q_vx1'], dummy_config_q_diag['q_vy1'],
        dummy_config_q_diag['q_vx2'], dummy_config_q_diag['q_vy2'],
        dummy_config_q_diag['q_theta'], dummy_config_q_diag['q_delta_theta']
    ]
    Qk_example = np.diag(Q_diag_values).astype(np.float32)


    stm_instance = StateTransitionModel(dt=1.0)
    ekf_predictor = EKFPredict(state_transition_model=stm_instance)

    # Example previous state and covariance
    x_hat_prev = StateVector.bbox_to_state([10, 20, 110, 120], initial_theta=0.1)
    # Set some initial velocities for testing
    x_hat_prev[StateVector.VX1] = 1.0
    x_hat_prev[StateVector.VY1] = 0.5
    x_hat_prev[StateVector.DELTA_THETA] = 0.02

    # Initial covariance P0 (example)
    P_prev_diag_values = [10.0] * StateVector.DIM # Simplified example
    P_prev = np.diag(P_prev_diag_values).astype(np.float32)

    print(f"Previous state estimate (x_hat_prev):\n{x_hat_prev}")
    print(f"\nPrevious covariance (P_prev):\n{P_prev[:3,:3]}... (showing top-left)") # Show part
    print(f"\nProcess Noise Qk:\n{Qk_example[:3,:3]}... (showing top-left)")

    predicted_x_minus, predicted_P_minus = ekf_predictor.predict(x_hat_prev, P_prev, Qk_example)

    print(f"\nPredicted state (x_k_minus):\n{predicted_x_minus}")
    # x1 should be 10 + 1*1 = 11
    # y1 should be 20 + 0.5*1 = 20.5
    # theta should be 0.1 + 0.02*1 = 0.12
    assert np.isclose(predicted_x_minus[StateVector.X1], 11.0)
    assert np.isclose(predicted_x_minus[StateVector.Y1], 20.5)
    assert np.isclose(predicted_x_minus[StateVector.THETA], 0.12)
    
    print(f"\nPredicted covariance (P_k_minus):\n{predicted_P_minus[:3,:3]}... (showing top-left)")
    assert predicted_P_minus.shape == (StateVector.DIM, StateVector.DIM)
    # Check if covariance increased (diagonal elements) due to Q and propagation
    assert predicted_P_minus[0,0] > P_prev[0,0] or np.isclose(Qk_example[0,0],0) # If Q[0,0] is not zero