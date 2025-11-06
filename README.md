# 🧠 Text Extractor using Deep Learning

This project is a **Flask-based web application** that performs **text extraction from images** using a **pre-trained deep learning model** .  
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

### 1️Clone the Repository
git clone https://github.com/ankurmishra48/TextExtractor.git
cd TextExtractor


##Install Required Packages
Run the following commands one by one to install all necessary dependencies:
pip install flask
pip install opencv-python
pip install tensorflow==2.11.0
pip install numpy==1.23.5
pip install protobuf==3.20.3
 


###Run the Application
python app.py
