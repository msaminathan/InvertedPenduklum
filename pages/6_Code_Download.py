import streamlit as st
import os
import zipfile
import io

st.set_page_config(page_title="Code Download", page_icon="💾", layout="wide")

st.title("💾 Code Download")

st.markdown("""
## Download Complete Source Code

You can download the entire source code for this application as a ZIP file.
""")

# Get the project directory (parent of pages directory)
import pathlib
project_dir = str(pathlib.Path(__file__).parent.parent)

def create_zip():
    """Create a ZIP file of the project"""
    zip_buffer = io.BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # List of files to include
        files_to_include = [
            'app.py',
            'requirements.txt',
            'README.md'
        ]
        
        # Add main files
        for file in files_to_include:
            file_path = os.path.join(project_dir, file)
            if os.path.exists(file_path):
                zip_file.write(file_path, file)
        
        # Add pages directory
        pages_dir = os.path.join(project_dir, 'pages')
        if os.path.exists(pages_dir):
            for root, dirs, files in os.walk(pages_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, project_dir)
                    zip_file.write(file_path, arcname)
    
    zip_buffer.seek(0)
    return zip_buffer

# Create download button
try:
    zip_buffer = create_zip()
    st.download_button(
        label="📥 Download Complete Source Code (ZIP)",
        data=zip_buffer,
        file_name="InvertedDoublePendulum.zip",
        mime="application/zip"
    )
except Exception as e:
    st.error(f"Error creating ZIP file: {str(e)}")
    st.info("You can manually download the files from the project directory.")

st.markdown("---")

st.header("Project Structure")

st.markdown("""
```
InvertedPendulum/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
└── pages/                          # Streamlit pages
    ├── 1_System_Diagram.py        # System diagram and visualization
    ├── 2_Dynamics.py              # Dynamics equations and analysis
    ├── 3_Control.py                # Control equations and LQR design
    ├── 4_Visualizations.py         # Plots and simulations
    ├── 5_References.py             # References and citations
    ├── 6_Code_Download.py          # This page
    └── 7_Cursor_AI.py              # Cursor AI description
```
""")

st.markdown("---")

st.header("Installation Instructions")

st.markdown("""
### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Steps

1. **Extract the ZIP file** to your desired location

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
""")

st.markdown("---")

st.header("Dependencies")

st.markdown("""
The following Python packages are required:

- **streamlit**: Web application framework
- **numpy**: Numerical computing
- **matplotlib**: Plotting and visualization
- **scipy**: Scientific computing (ODE solvers, LQR)
- **sympy**: Symbolic mathematics (optional, for equation display)

All dependencies are listed in `requirements.txt`.
""")

st.markdown("---")

st.header("Usage")

st.markdown("""
### Running the Application

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

### Customization

You can modify:
- System parameters (masses, lengths)
- Initial conditions
- Control gains
- Visualization settings
- Add new pages or features
""")

st.markdown("---")

st.header("License and Usage")

st.markdown("""
This code is provided for educational purposes. Feel free to:

- ✅ Use for learning and education
- ✅ Modify and adapt for your projects
- ✅ Share with attribution

### Attribution

If you use this code in your work, please acknowledge:
- Original source
- Educational purpose
- Any modifications made

### Disclaimer

This code is provided "as is" without warranty. For production control systems, 
ensure proper testing, validation, and safety measures.
""")

st.markdown("---")

st.header("GitHub Repository")

st.markdown("""
If you'd like to contribute or report issues, you can:

1. Fork the repository
2. Make your changes
3. Submit a pull request

For issues or questions, please open an issue on the repository.
""")

# Display file contents preview
st.markdown("---")
st.header("File Contents Preview")

file_to_preview = st.selectbox(
    "Select a file to preview",
    ["app.py", "requirements.txt", "README.md", "pages/1_System_Diagram.py"]
)

if file_to_preview:
    file_path = os.path.join(project_dir, file_to_preview)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            content = f.read()
        st.code(content, language='python')
    else:
        st.warning(f"File {file_to_preview} not found")

