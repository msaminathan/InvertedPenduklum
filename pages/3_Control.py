import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve_continuous_are

st.set_page_config(page_title="Control", page_icon="🎮", layout="wide")

st.title("🎮 Control Equations")

st.markdown("""
## Control Objective

The goal is to stabilize the inverted double pendulum at the upright equilibrium position:
- $x = 0$ (cart at origin)
- $\theta_1 = 0$ (bottom arm vertical)
- $\theta_2 = 0$ (top arm vertical)

This is achieved by applying a control force $F$ to the cart.
""")

st.header("Linear Quadratic Regulator (LQR) Control")

st.markdown("""
For the linearized system, we use LQR control to design a state feedback controller:

**$u = -\mathbf{K} \mathbf{x}$**

where $\mathbf{K}$ is the feedback gain matrix obtained by minimizing the cost function:

**$J = \int_0^{\infty} (\mathbf{x}^T \mathbf{Q} \mathbf{x} + u^T R u) dt$**
""")

st.markdown("### LQR Design")

st.latex(r"""
\mathbf{K} = R^{-1} \mathbf{B}^T \mathbf{P}
""")

st.markdown("where $\mathbf{P}$ is the solution to the Algebraic Riccati Equation (ARE):")

st.latex(r"""
\mathbf{A}^T \mathbf{P} + \mathbf{P} \mathbf{A} - \mathbf{P} \mathbf{B} R^{-1} \mathbf{B}^T \mathbf{P} + \mathbf{Q} = \mathbf{0}
""")

st.markdown("### State Feedback Control Law")

st.latex(r"""
F = -K_1 x - K_2 \theta_1 - K_3 \theta_2 - K_4 \dot{x} - K_5 \dot{\theta}_1 - K_6 \dot{\theta}_2
""")

st.markdown("---")

st.header("Interactive LQR Design")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("System Parameters")
    
    M = st.slider("Cart Mass (M)", 0.5, 5.0, 1.0, 0.1)
    m1 = st.slider("Bottom Mass (m₁)", 0.1, 2.0, 0.5, 0.1)
    m2 = st.slider("Top Mass (m₂)", 0.1, 2.0, 0.5, 0.1)
    L1 = st.slider("Bottom Arm Length (L₁)", 0.5, 2.0, 1.0, 0.1)
    L2 = st.slider("Top Arm Length (L₂)", 0.5, 2.0, 1.0, 0.1)
    g = 9.81
    
    st.subheader("LQR Weighting Matrices")
    
    # Q matrix weights
    q_x = st.slider("Q: x weight", 0.1, 100.0, 10.0, 0.1)
    q_theta1 = st.slider("Q: θ₁ weight", 0.1, 1000.0, 100.0, 0.1)
    q_theta2 = st.slider("Q: θ₂ weight", 0.1, 1000.0, 100.0, 0.1)
    q_xdot = st.slider("Q: ẋ weight", 0.1, 100.0, 1.0, 0.1)
    q_theta1dot = st.slider("Q: θ̇₁ weight", 0.1, 100.0, 1.0, 0.1)
    q_theta2dot = st.slider("Q: θ̇₂ weight", 0.1, 100.0, 1.0, 0.1)
    
    R = st.slider("R: Control weight", 0.1, 10.0, 1.0, 0.1)
    
    # Compute mass matrix M0
    M0 = np.array([
        [M + m1 + m2, m1*L1 + m2*L1, m2*L2],
        [m1*L1 + m2*L1, m1*L1**2 + m2*L1**2 + m2*L2**2 + 2*m2*L1*L2, 
         m2*L2**2 + m2*L1*L2],
        [m2*L2, m2*L2**2 + m2*L1*L2, m2*L2**2]
    ])
    
    # Compute G0 matrix
    G0 = np.array([
        [0, 0, 0],
        [0, (m1 + m2)*g*L1, m2*g*L2],
        [0, m2*g*L2, m2*g*L2]
    ])
    
    # Compute A matrix
    M0_inv = np.linalg.inv(M0)
    A_lower = -M0_inv @ G0
    A = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [A_lower, np.zeros((3, 3))]
    ])
    
    # Compute B matrix
    B_lower = M0_inv @ np.array([[1], [0], [0]])
    B = np.vstack([np.zeros((3, 1)), B_lower])
    
    # Q matrix
    Q = np.diag([q_x, q_theta1, q_theta2, q_xdot, q_theta1dot, q_theta2dot])
    
    # Solve ARE
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = (1/R) * B.T @ P
        
        st.success("LQR solution found!")
        
        st.subheader("LQR Gain Matrix K")
        st.write(f"K = [{K[0,0]:.2f}, {K[0,1]:.2f}, {K[0,2]:.2f}, "
                 f"{K[0,3]:.2f}, {K[0,4]:.2f}, {K[0,5]:.2f}]")
        
        # Display control law
        st.markdown("### Control Law")
        st.latex(f"""
        F = -{K[0,0]:.2f}x - {K[0,1]:.2f}\\theta_1 - {K[0,2]:.2f}\\theta_2 
        - {K[0,3]:.2f}\\dot{{x}} - {K[0,4]:.2f}\\dot{{\\theta}}_1 - {K[0,5]:.2f}\\dot{{\\theta}}_2
        """)
        
        # Check stability
        A_cl = A - B @ K
        eigvals = np.linalg.eigvals(A_cl)
        max_real = np.max(np.real(eigvals))
        
        if max_real < 0:
            st.success(f"✓ Closed-loop system is stable (max real part: {max_real:.4f})")
        else:
            st.error(f"✗ Closed-loop system is unstable (max real part: {max_real:.4f})")
        
        # Plot eigenvalues
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.real(eigvals), np.imag(eigvals), s=100, c='red', marker='x', linewidths=2)
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Real Part', fontsize=12)
        ax.set_ylabel('Imaginary Part', fontsize=12)
        ax.set_title('Closed-Loop Eigenvalues', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error computing LQR: {str(e)}")

with col2:
    st.subheader("Control Strategy")
    
    st.markdown("""
    ### LQR Control
    
    **Advantages:**
    - Optimal control for linearized system
    - Guaranteed stability (if system is controllable)
    - Systematic design procedure
    
    **Limitations:**
    - Only valid for small angles
    - Requires full state feedback
    - May need gain scheduling for larger angles
    """)
    
    st.markdown("""
    ### Alternative Methods
    
    1. **PID Control**: Simple but may not work well
    2. **Sliding Mode Control**: Robust to uncertainties
    3. **Backstepping**: For nonlinear systems
    4. **Energy Shaping**: Exploit energy structure
    5. **Model Predictive Control**: Handle constraints
    """)

st.markdown("---")

st.header("Nonlinear Control Considerations")

st.markdown("""
For larger angles, the linearized model is insufficient. Common approaches include:

### 1. Gain Scheduling
Switch between different LQR controllers based on the current state.

### 2. Feedback Linearization
Transform the nonlinear system into a linear one through coordinate transformation.

### 3. Lyapunov-Based Control
Design controllers that guarantee stability using Lyapunov functions.

### 4. Swing-Up Control
For large initial angles, use energy-based swing-up strategies before switching to LQR.
""")

st.markdown("---")

st.header("Control Implementation")

st.markdown("""
### Practical Considerations

1. **State Estimation**: If not all states are measurable, use observers (e.g., Kalman filter)
2. **Actuator Limits**: Saturate control input to physical limits
3. **Sampling**: Discrete-time implementation requires appropriate sampling rate
4. **Robustness**: Account for model uncertainties and disturbances
5. **Safety**: Implement emergency stops and bounds checking

### Control Architecture

```
Reference (x=0, θ₁=0, θ₂=0)
    ↓
State Feedback: u = -Kx
    ↓
Actuator (Cart Force F)
    ↓
System (Double Pendulum)
    ↓
Sensors (Position, Angles, Velocities)
    ↓
State Estimator (if needed)
    ↓
[Feedback Loop]
```
""")

