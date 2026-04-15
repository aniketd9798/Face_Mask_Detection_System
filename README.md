#  Real-Time Face Mask Detection and Analytics System

##  Overview

This project is a full-stack Computer Vision and Deep Learning system designed to detect whether a person is wearing a face mask in real time using a webcam.

Beyond detection, the system builds a complete data pipeline by logging mask compliance data and visualizing it through an interactive dashboard using Streamlit.

##  Key Features

###  Real-Time Detection

* Uses webcam input to detect faces instantly
* Classifies each face as:

  * ✅ Mask
  * ❌ No Mask
* Powered by OpenCV

###  High Accuracy Model

* Built using transfer learning
* Trained on a balanced dataset
* Developed with TensorFlow and Keras

###  Automated Data Logging

* Logs detection results into a CSV file
* Tracks masked and unmasked counts over time
* Maintains timestamp-based records

###  Live Analytics Dashboard

* Built with Streamlit
* Displays real-time statistics
* Shows compliance trends and comparisons

##  Technology Stack

*  Language: Python
*  Computer Vision: OpenCV
*  Deep Learning: TensorFlow, Keras
*  Data Processing: Pandas, NumPy
*  Web Dashboard: Streamlit

##  Project Structure

face-mask-detection-system/

├── detect_mask_webcam.py     #  Real-time detection and logging
├── app.py                    #  Streamlit dashboard
├── mask_detector.h5          #  Trained model
├── mask_analytics_log.csv    #  Logged data
├── requirements.txt          #  Dependencies
└── README.md                 #  Documentation


##  How to Run This Project Locally

###  Step 1: Clone the Repository

git clone https://github.com/aniketd9798/face-mask-detection-system.git
cd face-mask-detection-system

###  Step 2: Install Dependencies

pip install -r requirements.txt

###  Step 3: Run the Detection System

python detect_mask_webcam.py

 This will open your webcam, start real-time detection, and log data automatically.

###  Step 4: Run the Analytics Dashboard

(Open a new terminal)
streamlit run app.py

 This will launch the dashboard in your browser and display live analytics.

##  System Workflow

Webcam Input → Face Detection → Mask Classification → Data Logging → Dashboard Visualization

##  Use Cases

*  Schools and colleges
*  Offices and workplaces
*  Hospitals and healthcare environments
*  Public surveillance systems

##  Future Enhancements

*  Alert system for non-compliance
*  Cloud-based database integration
*  obile-friendly dashboard
*  Face recognition with identity tracking

##  Contribution

Contributions are welcome.
Fork the repository, create a feature branch, and submit a pull request.

##  License
This project is open-source and available under the MIT License.

