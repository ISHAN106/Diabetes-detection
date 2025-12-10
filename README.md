DiabetesDetector

A local prototype for predicting diabetes risk through a browser-based interface. The project includes:

A Flask backend (app.py) that loads a trained PyTorch model, preprocesses medical input features, and returns a diabetes prediction.

A simple HTML/CSS frontend (templates/) that allows a user to input health parameters and shows prediction results.

Scripts and utilities for training the model, saving weights, and performing local inference.

This README describes how to set up and run the project locally (Windows / PowerShell), required environment variables, troubleshooting tips, and where to look in the code.

Quick status

Frontend: Flask-rendered UI in templates/ (HTML forms → POST → prediction).

Backend: Flask app in app.py, uses PyTorch model loading, NumPy, Pandas, and Scikit-Learn preprocessing.

Model: A trained neural network stored as model/diabetes_model.pth (required for real predictions).

Dataset: Pima Indians Diabetes Dataset (UCI Repository).

Dependencies: Listed inside requirements.txt (recommended).

Prerequisites

Python 3.8+

pip (Python package installer)

Optional: Virtual environment for isolation

diabetes_model.pth available in the model/ folder
(Run train_model.py to retrain if file missing.)

Recommended development environment (Windows PowerShell)

Open one terminal for running the backend (Flask server).

Backend setup (Flask)

Create and activate a Python venv (optional but recommended):

cd C:\Users\ishan\Downloads\DiabetesDetector
python -m venv .venv
.\.venv\Scripts\Activate.ps1


Install Python dependencies:

pip install -r requirements.txt


(If requirements.txt is missing, install these manually:)

pip install flask numpy pandas torch scikit-learn


Start the backend:

python app.py


The backend runs on:
http://0.0.0.0:5000
 → Local access
Home page & form:
http://localhost:5000/

Key files and code locations

app.py — loads the model, receives form data, preprocesses values, returns predictions.

Endpoint /predict handles POST form input.

Uses scaling logic from preprocess.py if available.

train_model.py — train script for generating diabetes_model.pth.

Modify model layers and hyperparameters here.

templates/index.html — input form UI for health parameters (Glucose, BMI, Age, etc.)

templates/result.html — shows prediction response (Diabetic / Not Diabetic)

model/diabetes_model.pth — saved trained weights used at runtime.

Why "Diabetic" result appears even for low values?

The model prediction is based on historical correlations in the dataset.
Some combinations like:

Higher Glucose

High BMI

Higher Diabetes Pedigree Function

…can push the probability above threshold.
The threshold is typically 0.5 — you can tune this in app.py:

if prob >= 0.5:
    label = "Diabetic"


Modify threshold if needed.

Common issues & troubleshooting
❌ Server runs but page shows "Model not found"

Ensure model/diabetes_model.pth exists.

Re-train using:

python train_model.py

❌ Wrong Python / Torch errors

Install matching CPU version of PyTorch:

pip install torch --index-url https://download.pytorch.org/whl/cpu

❌ POST form gives 500 error

Check console logs printed by:

python app.py


Verify all fields exist:
Feature count must match 8 input parameters expected by model.

❌ No result page update

Ensure the form uses:

method="POST" action="/predict"


Restart Flask after editing templates.

Inspecting model output quickly

Test API using PowerShell:

Invoke-WebRequest -Uri "http://localhost:5000/" -Method GET


Debug logs are printed inside backend console whenever a prediction is made.

Development notes and next steps

Add better UI/validation for numeric fields (ranges, placeholders, tooltips)

Add graphical analytics dashboard showing risk scores over time

Deploy using Render / Railway / Azure Web App

Store predictions in MongoDB or SQLite for historical tracking

Add real-time probability meter instead of binary outcome

Where to look in the code for common edits
Requirement	File
Change model architecture	train_model.py
Change threshold for prediction	app.py
Change form UI labels	templates/index.html
Add new health features	Model + UI + preprocessing
Contributing

Please submit issues or pull requests with improved preprocessing, UI or model accuracy.

If you modify the dataset or feature count, update the model and HTML fields accordingly.

License

This repository is provided as-is for educational and prototyping purposes.
Not intended for real medical advice — always consult a physician for health decisions.

Contact / Help

If you'd like me to:

Add screenshots of UI → say "Add UI preview"

Create proper dataset download script → say "Add dataset script"

Add classified output logs or analytics → say "Add dashboard features"
