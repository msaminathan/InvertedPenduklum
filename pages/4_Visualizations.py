import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.linalg import solve_continuous_are

st.set_page_config(page_title="Visualizations", page_icon="📊", layout="wide")

st.title("📊 Visualizations and Simulations")

st.markdown("""
This page provides interactive visualizations and simulations of the inverted double pendulum system.
""")

# System parameters
st.sidebar.header("System Parameters")
M = st.sidebar.slider("Cart Mass (M)", 0.5, 5.0, 1.0, 0.1)
m1 = st.sidebar.slider("Bottom Mass (m₁)", 0.1, 2.0, 0.5, 0.1)
m2 = st.sidebar.slider("Top Mass (m₂)", 0.1, 2.0, 0.5, 0.1)
L1 = st.sidebar.slider("Bottom Arm Length (L₁)", 0.5, 2.0, 1.0, 0.1)
L2 = st.sidebar.slider("Top Arm Length (L₂)", 0.5, 2.0, 1.0, 0.1)
g = 9.81

# Compute linearized system matrices
def compute_system_matrices(M, m1, m2, L1, L2, g):
    M0 = np.array([
        [M + m1 + m2, m1*L1 + m2*L1, m2*L2],
        [m1*L1 + m2*L1, m1*L1**2 + m2*L1**2 + m2*L2**2 + 2*m2*L1*L2, 
         m2*L2**2 + m2*L1*L2],
        [m2*L2, m2*L2**2 + m2*L1*L2, m2*L2**2]
    ])
    
    G0 = np.array([
        [0, 0, 0],
        [0, (m1 + m2)*g*L1, m2*g*L2],
        [0, m2*g*L2, m2*g*L2]
    ])
    
    M0_inv = np.linalg.inv(M0)
    A_lower = -M0_inv @ G0
    A = np.block([
        [np.zeros((3, 3)), np.eye(3)],
        [A_lower, np.zeros((3, 3))]
    ])
    
    B_lower = M0_inv @ np.array([[1], [0], [0]])
    B = np.vstack([np.zeros((3, 1)), B_lower])
    
    return A, B, M0, G0

A, B, M0, G0 = compute_system_matrices(M, m1, m2, L1, L2, g)

# Tab selection
tab1, tab2, tab3, tab4 = st.tabs(["Eigenvalue Analysis", "Time Response", "Phase Portrait", "Animation"])

with tab1:
    st.header("Eigenvalue Analysis")
    
    st.markdown("""
    The eigenvalues of the open-loop system matrix **A** determine the stability of the system.
    """)
    
    eigvals = np.linalg.eigvals(A)
    max_real = np.max(np.real(eigvals))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Open-Loop Eigenvalues")
        st.write("Eigenvalues:")
        for i, eig in enumerate(eigvals):
            st.write(f"λ_{i+1} = {eig:.4f}")
        
        if max_real > 0:
            st.error(f"✗ System is unstable (max real part: {max_real:.4f})")
        else:
            st.success(f"✓ System is stable (max real part: {max_real:.4f})")
    
    with col2:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(np.real(eigvals), np.imag(eigvals), s=150, c='blue', marker='o', 
                   edgecolors='black', linewidths=2, label='Open-Loop')
        ax.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax.set_xlabel('Real Part', fontsize=12)
        ax.set_ylabel('Imaginary Part', fontsize=12)
        ax.set_title('Eigenvalue Plot', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        st.pyplot(fig)
    
    # LQR design
    st.subheader("LQR Controller Design")
    Q_weight = st.slider("Q matrix weight (diagonal)", 1.0, 1000.0, 100.0, 1.0)
    R_weight = st.slider("R weight", 0.1, 10.0, 1.0, 0.1)
    
    Q = Q_weight * np.eye(6)
    R = R_weight
    
    try:
        P = solve_continuous_are(A, B, Q, R)
        K = (1/R) * B.T @ P
        A_cl = A - B @ K
        eigvals_cl = np.linalg.eigvals(A_cl)
        max_real_cl = np.max(np.real(eigvals_cl))
        
        st.success("LQR controller computed!")
        st.write("Closed-loop eigenvalues:")
        for i, eig in enumerate(eigvals_cl):
            st.write(f"λ_{i+1} = {eig:.4f}")
        
        if max_real_cl < 0:
            st.success(f"✓ Closed-loop system is stable (max real part: {max_real_cl:.4f})")
        else:
            st.error(f"✗ Closed-loop system is unstable (max real part: {max_real_cl:.4f})")
        
        fig2, ax2 = plt.subplots(figsize=(8, 6))
        ax2.scatter(np.real(eigvals), np.imag(eigvals), s=150, c='blue', marker='o', 
                   edgecolors='black', linewidths=2, label='Open-Loop', alpha=0.5)
        ax2.scatter(np.real(eigvals_cl), np.imag(eigvals_cl), s=150, c='red', marker='x', 
                   linewidths=3, label='Closed-Loop')
        ax2.axvline(x=0, color='k', linestyle='--', linewidth=1)
        ax2.axhline(y=0, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('Real Part', fontsize=12)
        ax2.set_ylabel('Imaginary Part', fontsize=12)
        ax2.set_title('Open-Loop vs Closed-Loop Eigenvalues', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        st.pyplot(fig2)
        
    except Exception as e:
        st.error(f"Error computing LQR: {str(e)}")

with tab2:
    st.header("Time Response Simulation")
    
    st.markdown("""
    Simulate the response of the linearized system to initial conditions.
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        x0 = st.slider("Initial x", -1.0, 1.0, 0.0, 0.01)
        theta1_0 = st.slider("Initial θ₁ (degrees)", -30, 30, 5, 1)
        theta2_0 = st.slider("Initial θ₂ (degrees)", -30, 30, 5, 1)
    
    with col2:
        xdot0 = st.slider("Initial ẋ", -1.0, 1.0, 0.0, 0.01)
        theta1dot_0 = st.slider("Initial θ̇₁ (deg/s)", -30, 30, 0, 1)
        theta2dot_0 = st.slider("Initial θ̇₂ (deg/s)", -30, 30, 0, 1)
    
    use_control = st.checkbox("Use LQR Control", value=True)
    
    if use_control:
        Q = 100 * np.eye(6)
        R = 1.0
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = (1/R) * B.T @ P
        except:
            K = np.zeros((1, 6))
            st.warning("Could not compute LQR gains")
    else:
        K = np.zeros((1, 6))
    
    # Convert angles to radians
    theta1_0_rad = np.deg2rad(theta1_0)
    theta2_0_rad = np.deg2rad(theta2_0)
    theta1dot_0_rad = np.deg2rad(theta1dot_0)
    theta2dot_0_rad = np.deg2rad(theta2dot_0)
    
    # Initial state
    x0_vec = np.array([x0, theta1_0_rad, theta2_0_rad, xdot0, theta1dot_0_rad, theta2dot_0_rad])
    
    # Time vector
    t = np.linspace(0, 10, 1000)
    
    # Closed-loop system
    A_cl = A - B @ K
    
    # Simulate
    def system_dynamics(x, t):
        return A_cl @ x
    
    x_sim = odeint(system_dynamics, x0_vec, t)
    
    # Plot results
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    
    axes[0, 0].plot(t, x_sim[:, 0], 'b-', linewidth=2)
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('x (m)')
    axes[0, 0].set_title('Cart Position')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(t, np.rad2deg(x_sim[:, 1]), 'r-', linewidth=2)
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('θ₁ (degrees)')
    axes[0, 1].set_title('Bottom Arm Angle')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].plot(t, np.rad2deg(x_sim[:, 2]), 'g-', linewidth=2)
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('θ₂ (degrees)')
    axes[1, 0].set_title('Top Arm Angle')
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].plot(t, x_sim[:, 3], 'b--', linewidth=2)
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('ẋ (m/s)')
    axes[1, 1].set_title('Cart Velocity')
    axes[1, 1].grid(True, alpha=0.3)
    
    axes[2, 0].plot(t, np.rad2deg(x_sim[:, 4]), 'r--', linewidth=2)
    axes[2, 0].set_xlabel('Time (s)')
    axes[2, 0].set_ylabel('θ̇₁ (deg/s)')
    axes[2, 0].set_title('Bottom Arm Angular Velocity')
    axes[2, 0].grid(True, alpha=0.3)
    
    axes[2, 1].plot(t, np.rad2deg(x_sim[:, 5]), 'g--', linewidth=2)
    axes[2, 1].set_xlabel('Time (s)')
    axes[2, 1].set_ylabel('θ̇₂ (deg/s)')
    axes[2, 1].set_title('Top Arm Angular Velocity')
    axes[2, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Control input
    if use_control:
        u = -K @ x_sim.T
        fig_u, ax_u = plt.subplots(figsize=(10, 4))
        ax_u.plot(t, u[0, :], 'k-', linewidth=2)
        ax_u.set_xlabel('Time (s)')
        ax_u.set_ylabel('Control Force F (N)')
        ax_u.set_title('Control Input')
        ax_u.grid(True, alpha=0.3)
        st.pyplot(fig_u)

with tab3:
    st.header("Phase Portrait")
    
    st.markdown("""
    Phase portraits show the trajectory of the system in state space.
    """)
    
    plot_var1 = st.selectbox("Variable 1 (x-axis)", 
                             ["x", "θ₁", "θ₂", "ẋ", "θ̇₁", "θ̇₂"], index=0)
    plot_var2 = st.selectbox("Variable 2 (y-axis)", 
                             ["x", "θ₁", "θ₂", "ẋ", "θ̇₁", "θ̇₂"], index=3)
    
    var_map = {"x": 0, "θ₁": 1, "θ₂": 2, "ẋ": 3, "θ̇₁": 4, "θ̇₂": 5}
    idx1 = var_map[plot_var1]
    idx2 = var_map[plot_var2]
    
    # Simulate multiple trajectories
    n_trajectories = st.slider("Number of trajectories", 5, 20, 10)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_trajectories):
        # Random initial conditions
        x0_rand = np.random.uniform(-0.5, 0.5)
        theta1_0_rand = np.deg2rad(np.random.uniform(-10, 10))
        theta2_0_rand = np.deg2rad(np.random.uniform(-10, 10))
        xdot0_rand = np.random.uniform(-0.5, 0.5)
        theta1dot_0_rand = np.deg2rad(np.random.uniform(-10, 10))
        theta2dot_0_rand = np.deg2rad(np.random.uniform(-10, 10))
        
        x0_vec = np.array([x0_rand, theta1_0_rand, theta2_0_rand, 
                          xdot0_rand, theta1dot_0_rand, theta2dot_0_rand])
        
        # Use LQR control
        Q = 100 * np.eye(6)
        R = 1.0
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = (1/R) * B.T @ P
            A_cl = A - B @ K
        except:
            A_cl = A
        
        x_sim = odeint(lambda x, t: A_cl @ x, x0_vec, np.linspace(0, 5, 500))
        
        # Convert angles to degrees if needed
        if idx1 == 1 or idx1 == 2:
            var1 = np.rad2deg(x_sim[:, idx1])
        else:
            var1 = x_sim[:, idx1]
        
        if idx2 == 1 or idx2 == 2:
            var2 = np.rad2deg(x_sim[:, idx2])
        else:
            var2 = x_sim[:, idx2]
        
        ax.plot(var1, var2, alpha=0.6, linewidth=1.5)
        ax.plot(var1[0], var2[0], 'go', markersize=8)
        ax.plot(var1[-1], var2[-1], 'ro', markersize=8)
    
    ax.set_xlabel(plot_var1, fontsize=12)
    ax.set_ylabel(plot_var2, fontsize=12)
    ax.set_title(f'Phase Portrait: {plot_var1} vs {plot_var2}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(['Trajectory', 'Start', 'End'], loc='best')
    
    st.pyplot(fig)

with tab4:
    st.header("Animation")
    
    st.markdown("""
    Animated visualization of the pendulum motion.
    """)
    
    # Initial conditions
    x0_anim = st.slider("Initial x (animation)", -1.0, 1.0, 0.0, 0.01)
    theta1_0_anim = st.slider("Initial θ₁ (animation, degrees)", -30, 30, 5, 1)
    theta2_0_anim = st.slider("Initial θ₂ (animation, degrees)", -30, 30, 5, 1)
    
    use_control_anim = st.checkbox("Use LQR Control (animation)", value=True)
    
    # Convert to radians
    theta1_0_rad_anim = np.deg2rad(theta1_0_anim)
    theta2_0_rad_anim = np.deg2rad(theta2_0_anim)
    
    x0_vec_anim = np.array([x0_anim, theta1_0_rad_anim, theta2_0_rad_anim, 0, 0, 0])
    
    # Compute control
    if use_control_anim:
        Q = 100 * np.eye(6)
        R = 1.0
        try:
            P = solve_continuous_are(A, B, Q, R)
            K = (1/R) * B.T @ P
            A_cl = A - B @ K
        except:
            A_cl = A
    else:
        A_cl = A
    
    # Simulate
    t_anim = np.linspace(0, 5, 200)
    x_sim_anim = odeint(lambda x, t: A_cl @ x, x0_vec_anim, t_anim)
    
    # Create animation frames
    n_frames = st.slider("Number of frames", 10, 200, 50)
    frame_indices = np.linspace(0, len(t_anim)-1, n_frames, dtype=int)
    
    # Display frames
    cols = st.columns(4)
    for i, frame_idx in enumerate(frame_indices[:16]):  # Show first 16 frames
        with cols[i % 4]:
            fig_frame, ax_frame = plt.subplots(figsize=(3, 4))
            
            x_cart = x_sim_anim[frame_idx, 0]
            theta1 = x_sim_anim[frame_idx, 1]
            theta2 = x_sim_anim[frame_idx, 2]
            
            x1 = x_cart + L1 * np.sin(theta1)
            y1 = L1 * np.cos(theta1)
            x2 = x1 + L2 * np.sin(theta1 + theta2)
            y2 = y1 + L2 * np.cos(theta1 + theta2)
            
            ax_frame.set_xlim(-2, 2)
            ax_frame.set_ylim(-0.5, 2.5)
            ax_frame.set_aspect('equal')
            ax_frame.axhline(y=0, color='k', linewidth=2)
            
            # Draw cart
            cart_width = 0.2
            cart_height = 0.15
            cart_rect = plt.Rectangle((x_cart - cart_width/2, -cart_height/2), 
                                      cart_width, cart_height, 
                                      facecolor='gray', edgecolor='black')
            ax_frame.add_patch(cart_rect)
            
            # Draw arms
            ax_frame.plot([x_cart, x1], [0, y1], 'b-', linewidth=3)
            ax_frame.plot([x1, x2], [y1, y2], 'r-', linewidth=3)
            
            # Draw masses
            ax_frame.plot(x1, y1, 'bo', markersize=10)
            ax_frame.plot(x2, y2, 'ro', markersize=10)
            
            ax_frame.set_title(f't = {t_anim[frame_idx]:.2f}s', fontsize=8)
            ax_frame.axis('off')
            
            st.pyplot(fig_frame)
            plt.close(fig_frame)

