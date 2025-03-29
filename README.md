🌿 Plant Disease Recognition System

 📌 Overview
The Plant Disease Recognition System is a deep learning-based solution for detecting plant diseases from images. It uses a convolutional neural network (CNN) model to classify plant leaves into different disease categories, helping farmers and researchers take preventive measures.

 🚀 Features
- Deep Learning Model: Trained using CNNs for accurate disease classification.
- API Integration: Model deployed via Flask for easy accessibility.
- Real-Time Prediction: Upload an image to get instant disease classification.
- Data Analytics: Insights on disease occurrence and trends (future scope).

 📂 Project Structure
```
📦 Plant-Disease-Recognition-System
├── 📁 static               Contains images and static files
├── 📁 templates            HTML templates for the web app
├── 📁 models               Trained models (stored externally)
├── 📁 scripts              Data preprocessing and model training scripts
├── app.py                 Flask application
├── requirements.txt       Dependencies
└── README.md              Project documentation
```

 📸 Model & Dataset
- Dataset: Used publicly available plant disease datasets.
- Model: Trained using CNN architecture (MobileNetV2).
- Download Model: [Google Drive Link](https://drive.google.com/file/d/1_WQZv5MAuz3vP5ZOF_6EoPkjzEUitXZE/view?usp=sharing) 

 🔧 Installation & Setup
 Prerequisites
Ensure you have Python installed (>=3.8). Install dependencies using:
```bash
pip install -r requirements.txt
```

 Running the Flask App
```bash
python app.py
```
The app will run locally at `http://127.0.0.1:5000/`

 📊 Future Enhancements
- Add a dashboard for disease insights.
- Implement real-time image capture for predictions.
- Extend model to more plant species.

 🤝 Contributing
Feel free to fork this repository and contribute! Create a pull request with your enhancements or bug fixes.


---
📧 For queries, reach out at amanjain0411@gmail.com

🌱 Detect plant diseases early, save crops, and ensure food security!
