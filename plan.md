# Project Plan: Stage 3 - Medical Digital Twin Construction (SMPL-Based)

## 📌 Project Context & Goal
We have successfully completed **Stage 2 (Prediction & Interpretation)**, where we trained a Random Forest model to predict diabetes risk and integrated SHAP to explain feature importance. 
We are now entering **Stage 3 (Digital Twin Representation & What-If Simulation)**. The goal is to construct a 3D Patient Digital Twin using the **SMPL (Skinned Multi-Person Linear Model)** framework. The 3D model must dynamically change its shape (obesity level) based on the patient's BMI and change its color/visual feedback based on the predicted diabetes risk.

---

## 🛠️ Tech Stack & Dependencies
- **Backend/Logic**: Python 3.10+, PyTorch (for SMPL forward pass)
- **3D Framework**: `smplx` (Official SMPL/SMPL-X library)
- **Visualization/Rendering**: `pyvista` (for local scientific rendering) or `trimesh` (for exporting meshes)
- **UI/Dashboard**: `streamlit` (Recommended for rapid prototyping)

---

## 🚀 Step-by-Step Execution Plan for Agent

### Task 1: Environment Setup & Model Assets Preparation
- [ ] **1.1 Dependency Installation**: Install required libraries via pip.
  ```bash
  pip install smplx torch pyvista streamlit trimesh numpy
  ```
- [ ] **1.2 SMPL Model Acquisition**: 
  - Instruct the user or automate the verification of the official SMPL model file (`basicModel_f_lbs_10_207_0_v1.0.0.pkl` or `basicModel_m_lbs_10_207_0_v1.0.0.pkl`).
  - Create a directory structure: `./models/smpl/`.

### Task 2: Implement the Data-to-Shape Mapping Logic
- [ ] **2.1 Mathematical Mapping**: Implement a function that scales the real-world **BMI** value into the SMPL **Beta_0** (the first principal component controlling body mass index/obesity).
  - *Formula*: \(\beta_0 = (BMI_{patient} - 22.0) \times 0.5\) (Adjust scale factors based on visual realism).
- [ ] **2.2 Color Mapping**: Implement a function that maps the Stage 2 **Risk Probability (0-100%)** to a hex color code.
  - High Risk (>70%): Crimson Red (`#FF4D4D`)
  - Moderate Risk (40%-70%): Warning Orange (`#FFA500`)
  - Low Risk (<40%): Healthy Green (`#2ECC71`)

### Task 3: Build the 3D Twin Generation Pipeline
- [ ] **3.1 SMPL Forward Pass**: Write a Python module `twin_generator.py` that accepts `bmi` and `risk_prob`, passes the shape parameters to the `smplx` layer, and extracts the 3D `vertices` and `faces`.
- [ ] **3.2 Mesh Compilation**: Convert the raw vertices and faces into a PyVista PolyData object or a Trimesh object for rendering.

### Task 4: Integrate Stage 2 Model with Stage 3 What-If UI
- [ ] **4.1 Streamlit Dashboard Setup**: Create a clean web interface (`app.py`).
- [ ] **4.2 Sidebar Control**: Add interactive HTML/Streamlit sliders for user inputs (e.g., BMI, Age, Glucose, HbA1c).
- [ ] **4.3 Real-time Pipeline Loop**:
  1. User adjusts the **BMI slider** in the UI.
  2. The UI captures the input and feeds it into the **Stage 2 Random Forest Model**.
  3. The model outputs an **updated diabetes risk percentage**.
  4. Pass the new BMI and updated risk into the **Stage 3 Twin Generator**.
  5. **Render the 3D model dynamically** on the right side of the screen using PyVista's background plotter or exporting to a GLTF component.

---

## 📄 Expected File Structure to Create
```text
├── models/
│   └── smpl/
│       └── basicModel_f_lbs_10_207_0_v1.0.0.pkl  # SMPL weights
├── src/
│   ├── stage2_predict.py                         # Pre-trained Random Forest & SHAP logic
│   └── twin_generator.py                         # SMPL mesh & color processing
├── app.py                                        # Streamlit Web UI Entry point
└── requirements.txt                              # Project dependencies
```

---

## 🎯 Next Immediate Action Item for Agent
Please start with **Task 1** and **Task 2**. Write a clean, modular Python script named `twin_generator.py` that implements the standard SMPL forward pass using PyTorch, maps a sample BMI value to the body shape, and uses PyVista to display the resulting 3D mesh. Let me know if you need me to provide the mock Stage 2 prediction output to test the pipeline.
