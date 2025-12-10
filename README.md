# DiabetesDetector

A local prototype for predicting diabetes risk through a browser-based interface. The project includes:

- A **Flask backend** (`app.py`) that loads a trained PyTorch model, preprocesses medical input features, and returns a diabetes prediction.
- A **simple HTML/CSS frontend** (`templates/`) that allows a user to input health parameters and shows prediction results.
- Scripts and utilities for **training the model**, saving weights, and performing local inference.

This README describes how to set up and run the project locally (Windows / PowerShell), required dependencies, common troubleshooting steps, and where to look in the code.

---

## Quick status

- **Frontend:** Flask-rendered UI in `templates/` (HTML forms → POST → prediction).
- **Backend:** Flask app in `app.py`, uses PyTorch model loading, NumPy, Pandas, and Scikit-Learn preprocessing.
- **Model:** Trained neural network expected at `model/diabetes_model.pth` (required if you want predictions to work).
- **Dataset:** Pima Indians Diabetes Dataset (UCI Repository).
- **Dependencies:** Recommended to store in `requirements.txt`.

---

## Prerequisites

- Python 3.8+
- pip (Python package manager)
- Optional: Virtual environment for isolation
- A trained model saved as `model/diabetes_model.pth`  
  (You can generate this by running `train_model.py`.)

---

## Recommended development environment (Windows PowerShell)

- Open one PowerShell terminal for the backend (Flask server).

---

## Backend setup (Flask)

### 1. Create and activate a Python virtual environment (optional but recommended)

```powershell
cd C:\Users\ishan\Downloads\DiabetesDetector
python -m venv .venv
.\.venv\Scripts\Activate.ps1
2. Install Python dependencies
If you have a requirements.txt:

powershell
Copy code
pip install -r requirements.txt
If not, install the common dependencies manually:

powershell
Copy code
pip install flask numpy pandas torch scikit-learn
3. Start the backend
powershell
Copy code
python app.py
The backend runs on:

Base URL: http://0.0.0.0:5000 (internally)

Localhost URL: http://localhost:5000

The home page should display the form UI for entering medical parameters.

Project structure
A typical layout for this project:

text
Copy code
DiabetesDetector/
├── app.py                  # Flask backend (routes, prediction logic)
├── train_model.py          # Script to train and save the model
├── preprocess.py           # Optional: scaling and preprocessing utilities
├── model/
│   └── diabetes_model.pth  # Saved PyTorch model weights
├── templates/
│   ├── index.html          # Input form page
│   └── result.html         # Result display page
├── static/                 # (Optional) CSS, JS, images
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
Key files and what they do
app.py

Loads the trained model (diabetes_model.pth).

Defines routes:

GET / → renders the main form (index.html).

POST /predict → receives form data, preprocesses it, runs model prediction, and returns result page.

Applies any preprocessing (e.g., scaling) either inline or via preprocess.py.

train_model.py

Loads the Pima Indians Diabetes Dataset (CSV or from a path).

Splits data into train/test sets.

Builds and trains a PyTorch model.

Saves the model weights to model/diabetes_model.pth.

templates/index.html

Form-based UI for health parameters such as:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Sends data using method="POST" to /predict.

templates/result.html

Displays whether the predicted class is:

"Diabetic"

or "Not Diabetic"

Optionally shows the prediction probability and input summary.

Why "Diabetic" shows up for some inputs
The prediction depends on the probability output by the model.
Typically:

python
Copy code
if prob >= 0.5:
    label = "Diabetic"
else:
    label = "Not Diabetic"
If certain combinations of values (e.g., high glucose, high BMI, certain pedigree values) exceed this threshold, the model will label the user as "Diabetic".

You can change the threshold in app.py to make the model more or less strict:

python
Copy code
THRESHOLD = 0.6  # for example

if prob >= THRESHOLD:
    label = "Diabetic"
else:
    label = "Not Diabetic"
Common issues & troubleshooting
1. Server runs but shows an error about missing model
Symptom: Console or logs say model file not found.

Fix:

Make sure model/diabetes_model.pth exists.

Re-train and save the model using:

powershell
Copy code
python train_model.py
Confirm that app.py points to the correct path:

python
Copy code
MODEL_PATH = "model/diabetes_model.pth"
2. PyTorch or version errors when importing torch
Symptom: Import error or CPU/GPU mismatch.

Fix:

Install a CPU-only compatible PyTorch version (for most local setups):

powershell
Copy code
pip install torch --index-url https://download.pytorch.org/whl/cpu
3. POST /predict returns HTTP 500
Symptom: Submitting the form gives a server error.

Checks:

Run Flask in the terminal and read the stack trace.

Ensure all form fields have proper name attributes that match what app.py expects.

Confirm the number and order of features matches what the model was trained on.

4. Page does not update after submitting form
Symptom: You click submit, but nothing seems to happen.

Fix:

Confirm your form tag looks something like:

html
Copy code
<form method="POST" action="/predict">
After editing templates, stop and restart Flask:

powershell
Copy code
Ctrl + C
python app.py
Quick testing from browser or PowerShell
Open in browser:
http://localhost:5000/

Basic check from PowerShell:

powershell
Copy code
Invoke-WebRequest -Uri "http://localhost:5000/" -Method GET
Logs in the terminal will show requests and any errors.

Development notes and next steps
Add input validation for each field (e.g., no negative values).

Display probability as a percentage (e.g., "Risk: 72%").

Add charts/analytics (e.g., past test history for a user).

Add user authentication if multiple people will use it.

Containerize the app using Docker for deployment.

Where to look in the code for common edits
Change model architecture or training logic:
train_model.py

Change prediction threshold or output text:
app.py

Change field labels and input types (number/text/slider):
templates/index.html

Style, fonts, layout:
templates/index.html, templates/result.html, and optionally static/ CSS files.

Contributing
If you add features (e.g., better preprocessing, more robust model, or a nicer UI), please:

Keep code modular and readable.

Update this README if setup or behavior changes.

If you change the number of input features:

Update:

Model definition and training (train_model.py)

Input parsing (app.py)

HTML form fields (index.html)
