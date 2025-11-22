from flask import Flask, render_template, request
import torch
import numpy as np
import joblib
from model import DiabetesModel

app = Flask(__name__)

# Load model
model = DiabetesModel()
model.load_state_dict(torch.load("diabetes_model.pth"))
model.eval()

# Load scaler
scaler = joblib.load("scaler.pkl")

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        try:
            inputs = [float(request.form[x]) for x in [
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ]]
            inputs = np.array(inputs).reshape(1, -1)
            scaled = scaler.transform(inputs)
            input_tensor = torch.tensor(scaled, dtype=torch.float32)

            with torch.no_grad():
                prediction = model(input_tensor)
                result = "Diabetic" if prediction.item() > 0.5 else "Not Diabetic"
        except Exception as e:
            result = f"Error: {e}"

    return render_template('index.html', result=result)

# 🟨 This part might be missing in your file
if __name__ == '__main__':
    app.run(debug=True)
