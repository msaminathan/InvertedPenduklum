import streamlit as st
import numpy as np
import sympy as sp
from sympy import symbols, Matrix, simplify, latex

st.set_page_config(page_title="Dynamics", page_icon="⚙️", layout="wide")

st.title("⚙️ Dynamics of Motion")

st.markdown("""
## Lagrangian Formulation

The equations of motion are derived using Lagrangian mechanics. The Lagrangian is defined as:

**L = T - V**

where:
- **T** is the kinetic energy
- **V** is the potential energy
""")

# Define symbols
x, theta1, theta2 = symbols('x theta_1 theta_2', real=True)
x_dot, theta1_dot, theta2_dot = symbols('dot{x} dot{theta_1} dot{theta_2}', real=True)
M, m1, m2, L1, L2, g = symbols('M m_1 m_2 L_1 L_2 g', real=True, positive=True)

st.markdown("### Generalized Coordinates")
st.latex(r"""
\begin{align}
q &= \begin{bmatrix} x \\ \theta_1 \\ \theta_2 \end{bmatrix} \\
\dot{q} &= \begin{bmatrix} \dot{x} \\ \dot{\theta}_1 \\ \dot{\theta}_2 \end{bmatrix}
\end{align}
""")

st.markdown("### Position Vectors")

st.latex(r"""
\begin{align}
\text{Cart position:} \quad & \mathbf{r}_0 = x \hat{\mathbf{i}} \\
\text{Bottom mass position:} \quad & \mathbf{r}_1 = x \hat{\mathbf{i}} + L_1 \sin\theta_1 \hat{\mathbf{i}} + L_1 \cos\theta_1 \hat{\mathbf{j}} \\
\text{Top mass position:} \quad & \mathbf{r}_2 = \mathbf{r}_1 + L_2 \sin(\theta_1 + \theta_2) \hat{\mathbf{i}} + L_2 \cos(\theta_1 + \theta_2) \hat{\mathbf{j}}
\end{align}
""")

st.markdown("### Velocity Vectors")

st.latex(r"""
\begin{align}
\dot{\mathbf{r}}_0 &= \dot{x} \hat{\mathbf{i}} \\
\dot{\mathbf{r}}_1 &= \dot{x} \hat{\mathbf{i}} + L_1 \dot{\theta}_1 (\cos\theta_1 \hat{\mathbf{i}} - \sin\theta_1 \hat{\mathbf{j}}) \\
\dot{\mathbf{r}}_2 &= \dot{\mathbf{r}}_1 + L_2 (\dot{\theta}_1 + \dot{\theta}_2) [\cos(\theta_1 + \theta_2) \hat{\mathbf{i}} - \sin(\theta_1 + \theta_2) \hat{\mathbf{j}}]
\end{align}
""")

st.markdown("### Kinetic Energy")

st.latex(r"""
T = \frac{1}{2}M \dot{x}^2 + \frac{1}{2}m_1 |\dot{\mathbf{r}}_1|^2 + \frac{1}{2}m_2 |\dot{\mathbf{r}}_2|^2
""")

st.markdown("Expanding and simplifying:")

st.latex(r"""
\begin{align}
T &= \frac{1}{2}M \dot{x}^2 \\
&\quad + \frac{1}{2}m_1 [\dot{x}^2 + L_1^2 \dot{\theta}_1^2 + 2L_1 \dot{x} \dot{\theta}_1 \cos\theta_1] \\
&\quad + \frac{1}{2}m_2 [\dot{x}^2 + L_1^2 \dot{\theta}_1^2 + L_2^2 (\dot{\theta}_1 + \dot{\theta}_2)^2 \\
&\quad \quad + 2L_1 \dot{x} \dot{\theta}_1 \cos\theta_1 + 2L_2 \dot{x} (\dot{\theta}_1 + \dot{\theta}_2) \cos(\theta_1 + \theta_2) \\
&\quad \quad + 2L_1 L_2 \dot{\theta}_1 (\dot{\theta}_1 + \dot{\theta}_2) \cos\theta_2]
\end{align}
""")

st.markdown("### Potential Energy")

st.latex(r"""
V = m_1 g L_1 \cos\theta_1 + m_2 g [L_1 \cos\theta_1 + L_2 \cos(\theta_1 + \theta_2)]
""")

st.markdown("### Equations of Motion")

st.markdown("Using the Euler-Lagrange equations:")

st.latex(r"""
\frac{d}{dt}\left(\frac{\partial L}{\partial \dot{q}_i}\right) - \frac{\partial L}{\partial q_i} = Q_i
""")

st.markdown("where $Q_i$ are the generalized forces.")

st.markdown("#### Full Nonlinear Equations of Motion")

st.latex(r"""
\begin{align}
(M + m_1 + m_2)\ddot{x} + m_1 L_1 (\ddot{\theta}_1 \cos\theta_1 - \dot{\theta}_1^2 \sin\theta_1) \\
+ m_2 L_1 (\ddot{\theta}_1 \cos\theta_1 - \dot{\theta}_1^2 \sin\theta_1) \\
+ m_2 L_2 [(\ddot{\theta}_1 + \ddot{\theta}_2) \cos(\theta_1 + \theta_2) - (\dot{\theta}_1 + \dot{\theta}_2)^2 \sin(\theta_1 + \theta_2)] &= F
\end{align}
""")

st.latex(r"""
\begin{align}
m_1 L_1^2 \ddot{\theta}_1 + m_1 L_1 \ddot{x} \cos\theta_1 + m_2 L_1^2 \ddot{\theta}_1 \\
+ m_2 L_1 \ddot{x} \cos\theta_1 + m_2 L_2^2 (\ddot{\theta}_1 + \ddot{\theta}_2) \\
+ m_2 L_2 \ddot{x} \cos(\theta_1 + \theta_2) + m_2 L_1 L_2 \cos\theta_2 (2\ddot{\theta}_1 + \ddot{\theta}_2) \\
- m_2 L_1 L_2 \sin\theta_2 \dot{\theta}_2 (2\dot{\theta}_1 + \dot{\theta}_2) \\
+ m_1 g L_1 \sin\theta_1 + m_2 g L_1 \sin\theta_1 + m_2 g L_2 \sin(\theta_1 + \theta_2) &= 0
\end{align}
""")

st.latex(r"""
\begin{align}
m_2 L_2^2 (\ddot{\theta}_1 + \ddot{\theta}_2) + m_2 L_2 \ddot{x} \cos(\theta_1 + \theta_2) \\
+ m_2 L_1 L_2 \ddot{\theta}_1 \cos\theta_2 + m_2 L_1 L_2 \dot{\theta}_1^2 \sin\theta_2 \\
+ m_2 g L_2 \sin(\theta_1 + \theta_2) &= 0
\end{align}
""")

st.markdown("---")

st.header("Linearized Equations of Motion")

st.markdown("""
For small perturbations around the equilibrium point $(x, \\theta_1, \\theta_2) = (0, 0, 0)$, 
we linearize using the approximations:
""")

st.latex(r"""
\begin{align}
\sin\theta &\approx \theta \\
\cos\theta &\approx 1 \\
\dot{\theta}^2 &\approx 0 \quad \text{(second order terms)}
\end{align}
""")

st.markdown("### Linearized System Matrix Form")

st.latex(r"""
\mathbf{M}(\mathbf{q}) \ddot{\mathbf{q}} + \mathbf{C}(\mathbf{q}, \dot{\mathbf{q}}) \dot{\mathbf{q}} + \mathbf{G}(\mathbf{q}) = \mathbf{B} u
""")

st.markdown("For small angles, this becomes:")

st.latex(r"""
\mathbf{M}_0 \ddot{\mathbf{q}} + \mathbf{G}_0 \mathbf{q} = \mathbf{B} u
""")

st.markdown("where:")

st.latex(r"""
\mathbf{M}_0 = \begin{bmatrix}
M + m_1 + m_2 & m_1 L_1 + m_2 L_1 & m_2 L_2 \\
m_1 L_1 + m_2 L_1 & m_1 L_1^2 + m_2 L_1^2 + m_2 L_2^2 + 2m_2 L_1 L_2 & m_2 L_2^2 + m_2 L_1 L_2 \\
m_2 L_2 & m_2 L_2^2 + m_2 L_1 L_2 & m_2 L_2^2
\end{bmatrix}
""")

st.latex(r"""
\mathbf{G}_0 = \begin{bmatrix}
0 & 0 & 0 \\
0 & (m_1 + m_2) g L_1 & m_2 g L_2 \\
0 & m_2 g L_2 & m_2 g L_2
\end{bmatrix}
""")

st.latex(r"""
\mathbf{B} = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad u = F
""")

st.markdown("### State-Space Representation")

st.markdown("Defining the state vector:")

st.latex(r"""
\mathbf{x} = \begin{bmatrix} x \\ \theta_1 \\ \theta_2 \\ \dot{x} \\ \dot{\theta}_1 \\ \dot{\theta}_2 \end{bmatrix}
""")

st.markdown("The linearized state-space form is:")

st.latex(r"""
\dot{\mathbf{x}} = \mathbf{A} \mathbf{x} + \mathbf{B} u
""")

st.latex(r"""
\mathbf{A} = \begin{bmatrix}
\mathbf{0}_{3 \times 3} & \mathbf{I}_{3 \times 3} \\
-\mathbf{M}_0^{-1} \mathbf{G}_0 & \mathbf{0}_{3 \times 3}
\end{bmatrix}
""")

st.latex(r"""
\mathbf{B} = \begin{bmatrix}
\mathbf{0}_{3 \times 1} \\
\mathbf{M}_0^{-1} \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}
\end{bmatrix}
""")

st.markdown("---")

st.header("System Properties")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Characteristics
    
    1. **Underactuated**: 3 DOF, 1 control input
    2. **Unstable**: Upright equilibrium is unstable
    3. **Non-minimum phase**: Some zeros in right half-plane
    4. **Nonlinear**: Full dynamics are highly nonlinear
    5. **Coupling**: Strong coupling between all states
    """)

with col2:
    st.markdown("""
    ### Controllability
    
    The system is controllable if the controllability matrix:
    
    $\mathcal{C} = [\mathbf{B} \quad \mathbf{A}\mathbf{B} \quad \mathbf{A}^2\mathbf{B} \quad \ldots \quad \mathbf{A}^{5}\mathbf{B}]$
    
    has full rank (rank = 6).
    
    For typical parameter values, the system is controllable.
    """)

st.markdown("---")

st.markdown("""
## Notes

- The linearized model is valid only for small angles ($|\\theta_1|, |\\theta_2| < 10°$)
- For larger angles, the full nonlinear equations must be used
- The system exhibits complex dynamics including chaos for certain parameter ranges
- Control design typically uses the linearized model for initial design, then verified with nonlinear simulation
""")

