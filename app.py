import streamlit as st

st.set_page_config(
    page_title="Inverted Double Pendulum Dynamics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 Inverted Double Pendulum Dynamics")
st.markdown("""
Welcome to the interactive study of inverted double pendulum dynamics. 
Navigate through the pages using the sidebar to explore:
- System diagram and configuration
- Dynamics equations and analysis
- Control strategies
- Visualizations and simulations
- References and code download
""")

st.sidebar.success("Select a page from above ☝️")

st.markdown("""
## Overview

This application provides a comprehensive study of the inverted double pendulum system, 
which consists of:

- **Bottom Arm**: Pivoted to a frictionless cart that moves horizontally
- **Top Arm**: Pivoted to the bottom arm, free to rotate
- **Point Masses**: Located at the ends of each arm
- **Control**: Achieved through manipulation of the cart position

The system is inherently unstable and requires active control to maintain equilibrium.
""")

