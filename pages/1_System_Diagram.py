import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import matplotlib.patches as mpatches

st.set_page_config(page_title="System Diagram", page_icon="📐", layout="wide")

st.title("📐 System Diagram")

st.markdown("""
## System Configuration

The inverted double pendulum system consists of:

1. **Cart**: A frictionless cart that can move horizontally along the x-axis
2. **Bottom Arm**: A rigid arm of length L₁ with point mass m₁ at its end, pivoted to the cart
3. **Top Arm**: A rigid arm of length L₂ with point mass m₂ at its end, pivoted to the bottom arm

### System Parameters:
- **Cart position**: x (horizontal displacement)
- **Bottom arm angle**: θ₁ (measured from vertical)
- **Top arm angle**: θ₂ (measured from vertical, relative to bottom arm)
- **Cart mass**: M
- **Bottom arm mass**: m₁
- **Top arm mass**: m₂
- **Bottom arm length**: L₁
- **Top arm length**: L₂
""")

# Create interactive diagram
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("System Visualization")
    
    # Parameters for visualization
    L1 = st.slider("Bottom Arm Length (L₁)", 0.5, 2.0, 1.0, 0.1)
    L2 = st.slider("Top Arm Length (L₂)", 0.5, 2.0, 1.0, 0.1)
    theta1 = st.slider("Bottom Arm Angle θ₁ (degrees)", -180, 180, 0, 1)
    theta2 = st.slider("Top Arm Angle θ₂ (degrees)", -180, 180, 0, 1)
    cart_pos = st.slider("Cart Position x", -2.0, 2.0, 0.0, 0.1)
    
    # Convert angles to radians
    theta1_rad = np.deg2rad(theta1)
    theta2_rad = np.deg2rad(theta2)
    
    # Calculate positions
    x_cart = cart_pos
    y_cart = 0
    
    # Bottom arm end position
    x1 = x_cart + L1 * np.sin(theta1_rad)
    y1 = y_cart + L1 * np.cos(theta1_rad)
    
    # Top arm end position
    x2 = x1 + L2 * np.sin(theta1_rad + theta2_rad)
    y2 = y1 + L2 * np.cos(theta1_rad + theta2_rad)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 3.5)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linewidth=2)
    ax.axvline(x=0, color='k', linewidth=2, linestyle='--', alpha=0.7)
    ax.set_xlabel('x (horizontal)', fontsize=12)
    ax.set_ylabel('y (vertical)', fontsize=12)
    ax.set_title('Inverted Double Pendulum System', fontsize=14, fontweight='bold')
    
    # Draw cart
    cart_width = 0.3
    cart_height = 0.2
    cart = Rectangle((x_cart - cart_width/2, y_cart - cart_height/2), 
                     cart_width, cart_height, 
                     facecolor='gray', edgecolor='black', linewidth=2)
    ax.add_patch(cart)
    
    # Draw wheels
    wheel_radius = 0.05
    wheel1 = Circle((x_cart - cart_width/3, y_cart - cart_height/2 - wheel_radius), 
                    wheel_radius, facecolor='black')
    wheel2 = Circle((x_cart + cart_width/3, y_cart - cart_height/2 - wheel_radius), 
                    wheel_radius, facecolor='black')
    ax.add_patch(wheel1)
    ax.add_patch(wheel2)
    
    # Draw bottom arm
    ax.plot([x_cart, x1], [y_cart, y1], 'b-', linewidth=3, label='Bottom Arm (L₁)')
    # Draw top arm
    ax.plot([x1, x2], [y1, y2], 'r-', linewidth=3, label='Top Arm (L₂)')
    
    # Draw pivot points
    ax.plot(x_cart, y_cart, 'ko', markersize=10, label='Cart Pivot')
    ax.plot(x1, y1, 'ko', markersize=10, label='Joint Pivot')
    
    # Draw point masses
    mass1 = Circle((x1, y1), 0.08, facecolor='blue', edgecolor='black', linewidth=2)
    mass2 = Circle((x2, y2), 0.08, facecolor='red', edgecolor='black', linewidth=2)
    ax.add_patch(mass1)
    ax.add_patch(mass2)
    
    # Add labels
    ax.text(x_cart, y_cart - 0.4, f'Cart (M)\nx = {cart_pos:.2f}', 
            ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax.text(x1 + 0.2, y1 + 0.2, f'm₁\nθ₁ = {theta1}°', 
            ha='left', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(x2 + 0.2, y2 + 0.2, f'm₂\nθ₂ = {theta2}°', 
            ha='left', fontsize=10, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
    
    # Add angle arcs
    arc_radius = 0.3
    arc1 = mpatches.Arc((x_cart, y_cart), arc_radius, arc_radius, 
                        angle=0, theta1=0, theta2=np.rad2deg(theta1_rad), 
                        color='blue', linewidth=2)
    ax.add_patch(arc1)
    
    if abs(theta1_rad + theta2_rad) > 0.01:
        arc2 = mpatches.Arc((x1, y1), arc_radius, arc_radius, 
                            angle=np.rad2deg(theta1_rad), 
                            theta1=0, theta2=np.rad2deg(theta2_rad), 
                            color='red', linewidth=2)
        ax.add_patch(arc2)
    
    ax.legend(loc='upper right', fontsize=10)
    ax.set_facecolor('white')
    
    st.pyplot(fig)

with col2:
    st.subheader("System Parameters")
    
    st.markdown("""
    ### Coordinate System
    - **Origin**: Cart pivot point
    - **x-axis**: Horizontal (positive to the right)
    - **y-axis**: Vertical (positive upward)
    
    ### Variables
    - **x**: Cart position
    - **θ₁**: Bottom arm angle from vertical
    - **θ₂**: Top arm angle relative to bottom arm
    
    ### Physical Properties
    - **M**: Cart mass
    - **m₁**: Bottom arm point mass
    - **m₂**: Top arm point mass
    - **L₁**: Bottom arm length
    - **L₂**: Top arm length
    - **g**: Gravitational acceleration
    
    ### Notes
    - All joints are frictionless
    - Cart moves only horizontally
    - Arms are rigid with point masses at ends
    - System is underactuated (only cart can be controlled)
    """)

st.markdown("---")
st.markdown("""
## System Description

The inverted double pendulum is a classic example of an underactuated, nonlinear, 
and unstable system. The system has three degrees of freedom (cart position x, 
bottom arm angle θ₁, top arm angle θ₂) but only one control input (cart force).

### Key Characteristics:

1. **Instability**: The upright equilibrium position is unstable without control
2. **Nonlinearity**: The dynamics are highly nonlinear
3. **Underactuation**: Only one control input for three degrees of freedom
4. **Chaotic Behavior**: Small perturbations can lead to large deviations

The control objective is to stabilize the system at the upright position (θ₁ = 0, θ₂ = 0) 
by manipulating the cart position through applied forces.
""")


