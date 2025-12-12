# Inverted Double Pendulum Dynamics Study

A comprehensive multipage Streamlit application for studying the dynamics and control of an inverted double pendulum system.

## Overview

This application provides an interactive platform to explore:

- **System Configuration**: Interactive diagram of the inverted double pendulum
- **Dynamics Analysis**: Complete equations of motion using Lagrangian mechanics
- **Control Design**: LQR control design with interactive tuning
- **Visualizations**: Eigenvalue analysis, time responses, phase portraits, and animations
- **References**: Academic references and resources
- **Code Download**: Complete source code download

## System Description

The inverted double pendulum consists of:

- **Cart**: A frictionless cart that moves horizontally along the x-axis
- **Bottom Arm**: Rigid arm of length L₁ with point mass m₁, pivoted to the cart
- **Top Arm**: Rigid arm of length L₂ with point mass m₂, pivoted to the bottom arm

The system has three degrees of freedom (cart position x, bottom arm angle θ₁, top arm angle θ₂) but only one control input (cart force F), making it an underactuated system.

## Features

### 1. System Diagram Page
- Interactive visualization of the pendulum system
- Real-time parameter adjustment
- Coordinate system representation
- Angle visualization

### 2. Dynamics Page
- Lagrangian formulation
- Full nonlinear equations of motion
- Linearized equations for small perturbations
- State-space representation
- System properties analysis

### 3. Control Page
- LQR control design
- Interactive controller tuning
- Stability analysis
- Eigenvalue visualization
- Control law display

### 4. Visualizations Page
- Eigenvalue analysis (open-loop and closed-loop)
- Time response simulations
- Phase portraits
- Animation frames
- Multiple trajectory visualization

### 5. References Page
- Academic references (books, papers, articles)
- Online resources
- Software tools
- Additional reading materials

### 6. Code Download Page
- Complete source code download
- Installation instructions
- Usage guidelines
- Project structure

### 7. Cursor AI Page
- Description of AI-assisted development process
- Development workflow
- Technologies used

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- FFmpeg (required for MP4 animation generation)
  - **Linux**: `sudo apt-get install ffmpeg` or `sudo yum install ffmpeg`
  - **macOS**: `brew install ffmpeg`
  - **Windows**: Download from https://ffmpeg.org/download.html

### Steps

1. **Clone or download the repository**

2. **Navigate to the project directory**:
   ```bash
   cd InvertedPendulum
   ```

3. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

5. **Run the application**:
   ```bash
   streamlit run app.py
   ```

6. **Open your browser** to the URL shown in the terminal (usually http://localhost:8501)

## Dependencies

### Python Packages
- **streamlit**: Web application framework
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing (ODE solvers, LQR)
- **sympy**: Symbolic mathematics (for equation display)

### System Requirements
- **FFmpeg**: Required for MP4 animation generation (must be installed separately)
  - See Prerequisites section above for installation instructions

## Project Structure

```
InvertedPendulum/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
└── pages/                          # Streamlit pages
    ├── 1_System_Diagram.py        # System diagram and visualization
    ├── 2_Dynamics.py              # Dynamics equations and analysis
    ├── 3_Control.py                # Control equations and LQR design
    ├── 4_Visualizations.py         # Plots and simulations
    ├── 5_References.py             # References and citations
    ├── 6_Code_Download.py          # Code download page
    └── 7_Cursor_AI.py              # Cursor AI description
```

## Usage

1. Start the Streamlit app using `streamlit run app.py`
2. Navigate through pages using the sidebar
3. Adjust parameters using sliders and controls
4. View visualizations and simulations

### Key Features

- **Interactive System Diagram**: Adjust parameters and see real-time visualization
- **Dynamics Analysis**: View equations of motion and linearized models
- **Control Design**: Design LQR controllers interactively
- **Visualizations**: Eigenvalue plots, time responses, phase portraits
- **Simulations**: Animated pendulum motion

## Mathematical Background

### Equations of Motion

The system dynamics are derived using Lagrangian mechanics:

**L = T - V**

where T is kinetic energy and V is potential energy.

The full nonlinear equations are complex and involve coupling between all states. For small angles, the system can be linearized around the equilibrium point (x=0, θ₁=0, θ₂=0).

### Control Design

The application uses Linear Quadratic Regulator (LQR) control for the linearized system. The control law is:

**u = -Kx**

where K is the feedback gain matrix obtained by solving the Algebraic Riccati Equation.

## Limitations

- The linearized model is valid only for small angles (|θ₁|, |θ₂| < 10°)
- For larger angles, the full nonlinear equations must be used
- The current implementation focuses on linearized control
- Real-time performance may vary with complex computations

## Future Enhancements

- Full nonlinear simulation
- Additional control methods (sliding mode, backstepping, etc.)
- Swing-up control strategies
- Data export functionality
- Performance optimization
- Mobile responsiveness improvements

## License

This code is provided for educational purposes. Feel free to use, modify, and share with attribution.

## Acknowledgments

This application was developed using Cursor AI, demonstrating AI-assisted development capabilities. See the "Cursor AI" page in the application for details on the development process.

## References

See the "References" page in the application for a comprehensive list of academic references, books, papers, and online resources related to inverted pendulum systems and control theory.

## Contact

For questions, suggestions, or contributions, please refer to the code repository or open an issue.

---

**Note**: This application is intended for educational purposes. For production control systems, consult with control engineering professionals and perform thorough testing and validation.



