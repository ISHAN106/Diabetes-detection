# 🩺 Diabetes Detection Web App (Flask + PyTorch)

A Machine Learning based web application that predicts the likelihood of diabetes using medical diagnostic parameters.  
The model is trained using the **Pima Indians Diabetes Dataset** and deployed through a user-friendly **Flask** interface.

---

## 🚀 Features

- 🔬 Deep Learning model built with **PyTorch**
- 🧹 Automated data preprocessing (scaling & cleaning)
- 🌐 Web app built using **Flask**
- 🧑‍⚕️ Simple form-based UI for entering medical values
- 📈 Model trained & evaluated with metrics (Accuracy, Confusion Matrix)
- 💾 Saved trained model for quick inference

---

## 📂 Project Structure

diabetes-detection-app/
│
├── static/ # CSS, images, UI assets (optional)
├── templates/
│ ├── index.html # Input form page
│ └── result.html # Prediction result page
│
├── data/
│ └── diabetes.csv # Pima dataset
│
├── model/
│ └── diabetes_model.pth # Saved PyTorch trained model
│
├── app.py # Flask app script
├── train_model.py # Model training code
├── preprocess.py # Scaling & preprocessing logic
├── requirements.txt # All dependencies
└── README.md # Documentation

yaml
Copy code

---

## 🧠 Dataset

📌 **Source**: PIMA Indians Diabetes Dataset  
- Rows: 768  
- Features: 8 medical predictors (e.g., Glucose, BMI, Age)
- Label: Diabetes outcome (0 = No, 1 = Yes)

This dataset is widely used in healthcare ML research.

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/your-username/diabetes-detection-app.git
cd diabetes-detection-app
2️⃣ Create Virtual Environment (recommended)
bash
Copy code
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # macOS / Linux
3️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
4️⃣ Train the Model (optional – already included)
If you wish to retrain:

bash
Copy code
python train_model.py
5️⃣ Run the Web App
bash
Copy code
python app.py
🌍 Visit in browser:

cpp
Copy code
http://127.0.0.1:5000/
🖥️ Usage
Enter your values in the form (Glucose, BMI, Insulin, etc.)

Click Predict

App displays:

“Diabetic” 🚨

or “Not Diabetic” 🟢

📊 Model Details
Framework: PyTorch

Model Type: Feed-Forward Neural Network

Activation: ReLU

Optimizer: Adam

Loss: Binary CrossEntropy

Evaluation Metrics:

Accuracy

Confusion Matrix

You can modify the model architecture inside train_model.py.

📋 Requirements
See ➜ requirements.txt
Example dependencies:

nginx
Copy code
Flask
numpy
pandas
scikit-learn
torch
matplotlib
🛡️ Disclaimer
This project is purely research & education oriented.
It is not a medical diagnostic tool and should not replace professional healthcare advice.

🤝 Contributing
Contributions are welcome!
Submit issues or pull requests to enhance the application.

🙌 Acknowledgements
Dataset provided by UCI Machine Learning Repository

Developed using PyTorch and Flask

📜 License
