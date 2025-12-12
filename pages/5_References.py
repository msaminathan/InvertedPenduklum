import streamlit as st

st.set_page_config(page_title="References", page_icon="📚", layout="wide")

st.title("📚 References")

st.markdown("""
## Academic References

### Books

1. **Spong, M. W., Hutchinson, S., & Vidyasagar, M.** (2006). *Robot Modeling and Control*. 
   John Wiley & Sons.
   - Comprehensive treatment of underactuated systems
   - Chapter on inverted pendulum systems

2. **Khalil, H. K.** (2015). *Nonlinear Control*. Pearson.
   - Nonlinear control theory
   - Feedback linearization techniques

3. **Slotine, J. J. E., & Li, W.** (1991). *Applied Nonlinear Control*. 
   Prentice Hall.
   - Lyapunov-based control design
   - Sliding mode control

### Journal Articles

4. **Furuta, K., Yamakita, M., & Kobayashi, S.** (1992). "Swing-up control of inverted pendulum using pseudo-state feedback." 
   *Proceedings of the Institution of Mechanical Engineers, Part I: Journal of Systems and Control Engineering*, 206(4), 263-269.

5. **Åström, K. J., & Furuta, K.** (2000). "Swinging up a pendulum by energy control." 
   *Automatica*, 36(2), 287-295.

6. **Spong, M. W.** (1995). "The swing up control problem for the acrobot." 
   *IEEE Control Systems Magazine*, 15(1), 49-55.

7. **Fantoni, I., & Lozano, R.** (2002). *Non-linear Control for Underactuated Mechanical Systems*. 
   Springer Science & Business Media.

### Conference Papers

8. **Block, D. J., Astrom, K. J., & Spong, M. W.** (2007). "The reaction wheel pendulum." 
   *Synthesis Lectures on Control and Mechatronics*, 1(1), 1-105.

9. **Bortoff, S. A.** (1992). "Pseudolinearization of the acrobot using spline functions." 
   *Proceedings of the 31st IEEE Conference on Decision and Control*, 593-598.

10. **Murray, R. M., & Hauser, J.** (1991). "A case study in approximate linearization: 
    The acrobot example." *Proceedings of the 1991 American Control Conference*, 3587-3592.

### Online Resources

11. **MIT OpenCourseWare - Underactuated Robotics**
    - Course: 6.832 Underactuated Robotics
    - Website: https://underactuated.mit.edu/
    - Excellent resource for underactuated systems including double pendulum

12. **Wikipedia - Inverted Pendulum**
    - https://en.wikipedia.org/wiki/Inverted_pendulum
    - General overview and references

13. **Control Systems Wiki**
    - Various articles on pendulum control
    - LQR control design examples

### Software and Simulation Tools

14. **MATLAB/Simulink**
    - Control System Toolbox for LQR design
    - Simulink for simulation

15. **Python Control Systems Library**
    - https://python-control.readthedocs.io/
    - Open-source control systems library

16. **SciPy**
    - Scientific computing library
    - ODE solvers, optimization, linear algebra

## Key Concepts Covered

### Lagrangian Mechanics
- Derivation of equations of motion
- Generalized coordinates
- Energy-based modeling

### Linear Control Theory
- State-space representation
- Controllability analysis
- LQR optimal control
- Pole placement

### Nonlinear Control
- Feedback linearization
- Lyapunov stability
- Energy shaping
- Swing-up strategies

### System Analysis
- Eigenvalue analysis
- Phase portraits
- Time response
- Stability margins

## Additional Reading

### Control Design
- **Ziegler-Nichols tuning** for PID controllers
- **Model Predictive Control (MPC)** for constrained systems
- **Robust control** for uncertain systems
- **Adaptive control** for parameter variations

### Advanced Topics
- **Chaos theory** in double pendulum systems
- **Machine learning** for control (reinforcement learning)
- **Hybrid control** strategies
- **Real-time implementation** considerations

## Citation Format

If you use this work, please cite appropriately. For academic use:

```
Inverted Double Pendulum Dynamics Study
[Your Name/Institution]
[Year]
```

## Contact and Contributions

This application is open for educational purposes. For questions, suggestions, or contributions, 
please refer to the code repository.

---

**Note**: This application is intended for educational purposes. For production control systems, 
consult with control engineering professionals and perform thorough testing and validation.
""")

st.markdown("---")

st.header("Useful Links")

col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    ### Control Theory
    - [Control Systems Lectures](https://www.youtube.com/playlist?list=PLUMWjy5jgHK1NC52DXXrriwihVrYZKqjk)
    - [MIT OCW Control Systems](https://ocw.mit.edu/courses/mechanical-engineering/2-14-analysis-and-design-of-feedback-control-systems-spring-2014/)
    - [IEEE Control Systems Society](https://www.ieeecss.org/)
    """)

with col2:
    st.markdown("""
    ### Software Tools
    - [Python Control Systems Library](https://python-control.readthedocs.io/)
    - [SciPy Documentation](https://docs.scipy.org/)
    - [Streamlit Documentation](https://docs.streamlit.io/)
    - [Matplotlib Gallery](https://matplotlib.org/stable/gallery/index.html)
    """)


