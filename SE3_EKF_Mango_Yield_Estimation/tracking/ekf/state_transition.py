import numpy as np
from .state_vector import StateVector

class StateTransitionModel:
    def __init__(self, dt=1.0):
        """
        Defines the state transition model for the EKF.
        Paper Eq. 2 provides Fk (Jacobian of state transition).
        The actual state transition function f(xk-1, uk) is typically:
        x_k = Fk * x_{k-1} for a linear system.
        For non-linear, x_k = f(x_{k-1}). Here, it appears velocities update positions.

        State: [x1, y1, x2, y2, vx1, vy1, vx2, vy2, theta, delta_theta]

        Args:
            dt (float): Time step between filter updates (e.g., 1.0 second).
        """
        self.dt = dt

    def predict_next_state(self, prev_state):
        """
        Predicts the next state x_k_minus = f(x_k_minus_1).
        This is the non-linear state transition function.
        x_new = x_old + v * dt
        theta_new = theta_old + delta_theta * dt

        Args:
            prev_state (np.array): Previous state vector x_{k-1}.

        Returns:
            np.array: Predicted state vector x_k^{-}.
        """
        predicted_state = prev_state.copy()

        # Update positions based on their velocities
        predicted_state[StateVector.X1] += prev_state[StateVector.VX1] * self.dt
        predicted_state[StateVector.Y1] += prev_state[StateVector.VY1] * self.dt
        predicted_state[StateVector.X2] += prev_state[StateVector.VX2] * self.dt
        predicted_state[StateVector.Y2] += prev_state[StateVector.VY2] * self.dt

        # Update angle based on angular velocity
        predicted_state[StateVector.THETA] += prev_state[StateVector.DELTA_THETA] * self.dt

        # Velocities and angular velocity are assumed constant in this simple model,
        # unless affected by process noise Q.
        # If there's a control input uk, it would be added here. Paper states uk in Eq.4.
        # For now, assuming uk is zero or absorbed into Q.

        return predicted_state

    def get_jacobian_Fk(self, prev_state_ignored=None):
        """
        Returns the Jacobian matrix Fk of the state transition function.
        For the defined `predict_next_state`, the system is linear, so Fk is constant.
        Paper Eq. 2:
        Fk = [I4x4  dt*I4x4  0_4x2]
             [0_4x4  I4x4     0_4x2]
             [0_2x4  0_2x4    M_angle]
        where M_angle = [1 dt; 0 1] for (theta, delta_theta)

        State: [x1,y1,x2,y2,  vx1,vy1,vx2,vy2,  theta,d_theta]
                (pos_tl,pos_br) (vel_tl,vel_br)  (angle_params)
        Indices:  0-3           4-7             8-9

        Fk structure:
           dx/dx  dx/dv  dx/d_ang
           dv/dx  dv/dv  dv/d_ang
           d_ang/dx d_ang/dv d_ang/d_ang

        df_pos/d_pos = I (4x4)
        df_pos/d_vel = dt * I (4x4 for vx1,vy1,vx2,vy2)
        df_pos/d_ang = 0 (4x2)

        df_vel/d_pos = 0 (4x4)
        df_vel/d_vel = I (4x4)
        df_vel/d_ang = 0 (4x2)

        df_ang/d_pos = 0 (2x4)
        df_ang/d_vel = 0 (2x4)
        df_ang/d_ang = [1 dt; 0 1] (2x2 for theta, delta_theta)
        
        Args:
            prev_state_ignored: Not used here as Fk is constant for this linear model.
                                 Included for EKF interface consistency.

        Returns:
            np.array: The 10x10 Jacobian matrix Fk.
        """
        F = np.eye(StateVector.DIM, dtype=np.float32)

        # Position updates from velocity
        # dx1/dvx1 = dt, dy1/dvy1 = dt, dx2/dvx2 = dt, dy2/dvy2 = dt
        F[StateVector.X1, StateVector.VX1] = self.dt
        F[StateVector.Y1, StateVector.VY1] = self.dt
        F[StateVector.X2, StateVector.VX2] = self.dt
        F[StateVector.Y2, StateVector.VY2] = self.dt

        # Angle update from angular velocity
        # dtheta/d_delta_theta = dt
        F[StateVector.THETA, StateVector.DELTA_THETA] = self.dt
        
        # This corresponds to the structure in the paper's Eq. 2,
        # assuming the state vector is ordered as pos_corners, vel_corners, angle_params.
        # My F matrix reflects:
        # x1_k = x1_{k-1} + vx1_{k-1}*dt  => dx1_k/dx1_{k-1}=1, dx1_k/dvx1_{k-1}=dt
        # vx1_k = vx1_{k-1}                => dvx1_k/dvx1_{k-1}=1
        # theta_k = theta_{k-1} + dtheta_{k-1}*dt => dtheta_k/dtheta_{k-1}=1, dtheta_k/d_dtheta_{k-1}=dt
        # dtheta_k = dtheta_{k-1}          => d_dtheta_k/d_dtheta_{k-1}=1

        return F

if __name__ == '__main__':
    dt_val = 1.0 # 1 second time step
    stm = StateTransitionModel(dt=dt_val)

    # Example previous state
    # [x1, y1, x2, y2, vx1, vy1, vx2, vy2, theta, delta_theta]
    prev_s = np.array([10, 20, 110, 120, 1, 0.5, 1.5, 0.8, 0.1, 0.02], dtype=np.float32)
    print(f"Previous state:\n{prev_s}")

    predicted_s = stm.predict_next_state(prev_s)
    print(f"\nPredicted next state (using f(x)) with dt={dt_val}:\n{predicted_s}")
    # Expected x1_new = 10 + 1*1.0 = 11
    # Expected y1_new = 20 + 0.5*1.0 = 20.5
    # Expected x2_new = 110 + 1.5*1.0 = 111.5
    # Expected y2_new = 120 + 0.8*1.0 = 120.8
    # Expected theta_new = 0.1 + 0.02*1.0 = 0.12
    # Velocities remain same.
    assert np.isclose(predicted_s[StateVector.X1], 11.0)
    assert np.isclose(predicted_s[StateVector.Y1], 20.5)
    assert np.isclose(predicted_s[StateVector.X2], 111.5)
    assert np.isclose(predicted_s[StateVector.Y2], 120.8)
    assert np.isclose(predicted_s[StateVector.VX1], 1.0) # Unchanged
    assert np.isclose(predicted_s[StateVector.THETA], 0.12)
    assert np.isclose(predicted_s[StateVector.DELTA_THETA], 0.02) # Unchanged

    Fk_matrix = stm.get_jacobian_Fk()
    print(f"\nJacobian Fk (for dt={dt_val}):\n{Fk_matrix}")
    
    # Check Fk structure (e.g., F[0,4] should be dt for x1 dependency on vx1)
    assert np.isclose(Fk_matrix[StateVector.X1, StateVector.VX1], dt_val)
    assert np.isclose(Fk_matrix[StateVector.X1, StateVector.X1], 1.0) # Identity part
    assert np.isclose(Fk_matrix[StateVector.VX1, StateVector.VX1], 1.0) # Identity part
    assert np.isclose(Fk_matrix[StateVector.THETA, StateVector.DELTA_THETA], dt_val)
    assert np.isclose(Fk_matrix[StateVector.THETA, StateVector.THETA], 1.0) # Identity part
    assert np.isclose(Fk_matrix[StateVector.DELTA_THETA, StateVector.DELTA_THETA], 1.0) # Identity part

    # For linear system x_k = Fk * x_{k-1}
    # This should match predict_next_state if system is indeed linear as modeled by Fk.
    predicted_s_linear = Fk_matrix @ prev_s
    print(f"\nPredicted next state (using Fk * x_prev):\n{predicted_s_linear}")
    assert np.allclose(predicted_s, predicted_s_linear), "Mismatch between non-linear prediction and linear Fk prediction"