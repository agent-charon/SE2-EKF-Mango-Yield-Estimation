import numpy as np
from .state_vector import StateVector

class MeasurementModel:
    # Measurement vector zk = [Cx, Cy, width, height, theta_obs]
    # Cx, Cy: center of the bounding box
    # width, height: width and height of the bounding box
    # theta_obs: observed rotational angle of the bounding box
    MEAS_DIM = 5 
    CX, CY, W, H, THETA_OBS = 0, 1, 2, 3, 4

    def __init__(self):
        pass

    def h(self, predicted_state_xk_minus):
        """
        Measurement function h(x_k^{-}).
        Converts a predicted state vector into the expected measurement vector.
        Paper Eq.3 defines zk = [Cx, Cy, x2-x1, y2-y1, theta]

        Args:
            predicted_state_xk_minus (np.array): Predicted state vector x_k^{-}.
                                                 [x1,y1,x2,y2,vx1,vy1,vx2,vy2,theta,d_theta]

        Returns:
            np.array: Expected measurement vector z_k_hat.
                      [Cx, Cy, width, height, theta_obs]
        """
        x1 = predicted_state_xk_minus[StateVector.X1]
        y1 = predicted_state_xk_minus[StateVector.Y1]
        x2 = predicted_state_xk_minus[StateVector.X2]
        y2 = predicted_state_xk_minus[StateVector.Y2]
        theta = predicted_state_xk_minus[StateVector.THETA]

        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        width = x2 - x1 # Assuming x2 > x1, y2 > y1. Can be abs() for robustness.
        height = y2 - y1

        # Ensure width and height are positive.
        # If state allows x1 > x2, then width/height could be negative.
        # A common convention is width = abs(x2-x1).
        # The paper's (x2-x1) might imply an ordering, or it's just for derivation.
        # Using abs() is safer for actual measurement comparison.
        # However, for Jacobian calculation, non-abs might be intended. Let's stick to paper's form.
        # If x1,y1 is top-left and x2,y2 is bottom-right, then x2-x1 and y2-y1 are positive.
        
        # The state vector definition (x1,y1,x2,y2) doesn't enforce TL/BR ordering strictly,
        # but it's implied. If vx1, vy1 are for TL and vx2, vy2 for BR, then (x2-x1) could become negative
        # if BR moves past TL. The EKF should handle this.
        # For now, assume x2 > x1 and y2 > y1 from typical detection outputs.

        return np.array([center_x, center_y, width, height, theta], dtype=np.float32)

    def get_jacobian_Hk(self, predicted_state_xk_minus_ignored=None):
        """
        Returns the Jacobian matrix Hk of the measurement function h.
        Hk = dh/dx | at x_k^{-}
        Measurement z = [Cx, Cy, W, H, theta_obs]
        State x = [x1,y1,x2,y2, vx1,vy1,vx2,vy2, theta,d_theta]

        Partial derivatives:
        dCx/dx1 = 0.5, dCx/dx2 = 0.5, others 0
        dCy/dy1 = 0.5, dCy/dy2 = 0.5, others 0
        dW/dx1 = -1,  dW/dx2 = 1,   others 0  (W = x2-x1)
        dH/dy1 = -1,  dH/dy2 = 1,   others 0  (H = y2-y1)
        dTheta_obs/dtheta = 1,      others 0

        Hk is a 5x10 matrix.
        Rows: Cx, Cy, W, H, Theta_obs
        Cols: x1,y1,x2,y2, vx1,vy1,vx2,vy2, theta,d_theta
        
        Args:
            predicted_state_xk_minus_ignored: Not used if Jacobian is constant.
        Returns:
            np.array: The 5x10 Jacobian matrix Hk.
        """
        H = np.zeros((self.MEAS_DIM, StateVector.DIM), dtype=np.float32)

        # Row for Cx = (x1+x2)/2
        H[self.CX, StateVector.X1] = 0.5
        H[self.CX, StateVector.X2] = 0.5

        # Row for Cy = (y1+y2)/2
        H[self.CY, StateVector.Y1] = 0.5
        H[self.CY, StateVector.Y2] = 0.5

        # Row for W = x2-x1
        H[self.W, StateVector.X1] = -1.0
        H[self.W, StateVector.X2] = 1.0

        # Row for H = y2-y1
        H[self.H, StateVector.Y1] = -1.0
        H[self.H, StateVector.Y2] = 1.0

        # Row for Theta_obs = theta (from state)
        H[self.THETA_OBS, StateVector.THETA] = 1.0
        
        # This matches paper's Eq.2 for Hk structure.

        return H

if __name__ == '__main__':
    mm = MeasurementModel()

    # Example predicted state
    pred_s = np.array([10, 20, 110, 120, 1, 0.5, 1.5, 0.8, 0.1, 0.02], dtype=np.float32)
    print(f"Predicted state (x_k_minus):\n{pred_s}")

    expected_measurement = mm.h(pred_s)
    print(f"\nExpected measurement (z_k_hat = h(x_k_minus)):\n{expected_measurement}")
    # Cx = (10+110)/2 = 60
    # Cy = (20+120)/2 = 70
    # W = 110-10 = 100
    # H = 120-20 = 100
    # Theta_obs = 0.1
    assert np.isclose(expected_measurement[mm.CX], 60)
    assert np.isclose(expected_measurement[mm.CY], 70)
    assert np.isclose(expected_measurement[mm.W], 100)
    assert np.isclose(expected_measurement[mm.H], 100)
    assert np.isclose(expected_measurement[mm.THETA_OBS], 0.1)

    Hk_matrix = mm.get_jacobian_Hk()
    print(f"\nJacobian Hk:\n{Hk_matrix}")
    assert Hk_matrix.shape == (mm.MEAS_DIM, StateVector.DIM)
    # Check some specific Jacobian values
    assert np.isclose(Hk_matrix[mm.CX, StateVector.X1], 0.5)
    assert np.isclose(Hk_matrix[mm.W, StateVector.X1], -1.0)
    assert np.isclose(Hk_matrix[mm.THETA_OBS, StateVector.THETA], 1.0)
    assert np.isclose(Hk_matrix[mm.CX, StateVector.VX1], 0.0) # No direct dependency of measurement on velocity