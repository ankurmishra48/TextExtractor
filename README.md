# 🧠 Text Extractor using Deep Learning

This project is a **Flask-based web application** that performs **text extraction from images** using a **pre-trained deep learning model** (`trained_model_13_3.h5`).  
It leverages **OpenCV** for image preprocessing and **TensorFlow/Keras** for model inference.

---

## 🚀 Features

- Upload image through a web interface  
- Preprocess the image using OpenCV filters  
- Predict the text content using a pre-trained LSTM-based model  
- Display or return the extracted text as output  

---

## 🧩 Tech Stack

| Component | Technology Used |
|------------|----------------|
| Backend Framework | Flask |
| Deep Learning | TensorFlow 2.11.0, Keras |
| Image Processing | OpenCV |
| Language | Python 3.9 / 3.10 |
| Frontend | HTML, CSS (Templates in `templates/` folder) |

---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/ankurmishra48/TextExtractor.git
cd TextExtractor
2️⃣ Install Required Packages
Run the following commands one by one to install all necessary dependencies:

bash
Copy code
pip install flask
pip install opencv-python
pip install tensorflow==2.11.0
pip install numpy==1.23.5
pip install protobuf==3.20.3
💡 Tip: It’s recommended to install them inside a virtual environment to avoid version conflicts.

3️⃣ Create and Activate Virtual Environment (Optional but Recommended)
bash
Copy code
python -m venv .venv_tf211
Activate it:

On Windows (PowerShell):

bash
Copy code
.\.venv_tf211\Scripts\activate
On macOS/Linux:

bash
Copy code
source .venv_tf211/bin/activate
Then repeat the dependency installation commands above inside this environment.

4️⃣ Run the Application
bash
Copy code
python app.py
