Real Time Face Mask Detection and Analytics System

Overview
This project is an end to end Computer Vision and Deep Learning application that detects in real time whether a person is wearing a face mask or not. It uses a custom trained Convolutional Neural Network for high accuracy predictions and OpenCV for live face tracking.

Going beyond standard detection, this system acts as a complete data pipeline by automatically logging compliance data to a CSV file and visualizing it in a real time web dashboard using Streamlit.

Key Features
Real Time Detection: Processes live webcam feeds to detect faces and classify mask usage instantly.
High Accuracy: Powered by a transfer learning model trained on a balanced dataset of masked and unmasked faces.
Automated Data Logging: Tracks the number of masked vs unmasked individuals over time and saves the data to a local CSV file.
Live Analytics Dashboard: A Streamlit web application that reads the live dataset and displays key metrics and compliance trend lines.

Technology Stack
Language: Python
Computer Vision: OpenCV
Deep Learning: TensorFlow and Keras
Data Processing: Pandas and NumPy
Web Dashboard: Streamlit

File Structure
detect_mask_webcam.py: The main OpenCV script that opens the webcam, detects faces, runs the AI model, and logs data.
app.py: The Streamlit dashboard script that visualizes the logged data.
mask_detector.h5: The pre trained Deep Learning model.
mask_analytics_log.csv: The live database tracking compliance over time.
requirements.txt: Required libraries to run the project.

How to Run This Project Locally

Step 1: Clone the repository
git clone https://github.com/yourusername/face-mask-detection-system.git
cd face-mask-detection-system

Step 2: Install dependencies
pip install -r requirements.txt

Step 3: Run the Detection Camera
This will open your webcam and start logging data.
python detect_mask_webcam.py

Step 4: Run the Analytics Dashboard
Open a new terminal window and run this command to see the live website.
streamlit run app.py
Step 4: Run the Analytics Dashboard
Open a new terminal window and run this command to see the live website.
streamlit run app.py](https://github.com/aniketd9798/Face_Mask_Detection_System/edit/main/README.md)
