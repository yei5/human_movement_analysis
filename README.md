# Human activity recognition
This project is a part of the  **Inteligencia artificial I** course in the Software engineering, Universidad Icesi, Cali Colombia. 

#### Project status: Active

## Contributing Members

**Team Leader: Yeison Antonio Rodriguez [github](https://github.com/yei5)**

**Instructor: Milton Sarria [github](https://github.com/miltonsarria)**

## Contact
* Feel free to contact the team leader or the instructor with any questions or if you are interested in contributing!


## Project Intro/Objective
The purpose of this project is to develop a tool that can analyze human movement in real time and automatically classify specific actions such as walking, turning, sitting, and standing up.
Using MediaPipe Pose for landmark detection, the system will extract numerical features (angles, inclinations, velocities) and feed them into a supervised learning model (SVM, Random Forest, or XGBoost).
The project aims to support applications in health monitoring, workplace safety, and posture assessment through an accessible and interpretable AI solution.

### Methods Used
* Data Collection and Annotation
* Feature Extraction from Pose Landmarks
* Machine Learning (SVM, Random Forest, XGBoost)
* Data Preprocessing and Normalization
* Model Evaluation (Accuracy, Precision, Recall, F1-score)
* Data Visualization (Confusion Matrices, ROC Curves)

### Technologies
* Python
* Github

## Project Description

This project addresses the task of recognizing human activities in real time using 2D/3D pose data. The process involves:

1. **Data Collection**: Gathering pose data streams using video or webcam with MediaPipe Pose.
2. **Annotation**: Labeling each example with the ground-truth activity (e.g., walking, sitting, standing).
3. **Feature Engineering**: Transforming raw pose landmarks into meaningful features (joint angles, inclination, and velocities).
4. **Preprocessing**: Cleaning, normalizing, and split into training/validation/testing sets.
5. **Modeling**: Training and tuning several ML classifiers.
6. **Evaluation**: Using common metrics (accuracy, precision, recall, F1-score, confusion matrix).
7. **Deployment**: Implementing a real-time detection tool using the trained model.

### Data Sources

- Pose landmarks extracted via MediaPipe Pose library
- Video recordings of individuals performing target activities and manually annotated.

### Key Questions and Hypotheses

- Which features (angles, inclination, etc.) most strongly contribute to distinguishing between activities?
- Can lightweight models (SVM, RF) perform as well as more complex models (XGBoost) for real-time recognition?
- How robust are models to variance in user body types and camera conditions?

### Challenges and Blockers

- Ensuring high-quality and consistent labels during annotation
- Handling noisy or missing pose landmarks in real-time streams
- Generalizing recognition to users and scenarios not seen during training
- Balancing accuracy with inference speed for real-time operation

## Getting Started
Instructions for contributors
1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept offline, if you need access to it please contact the team leader.
    
3. Data processing/transformation scripts are being kept [here](entrega1/src/data/). Processed data is stored in the entrega1/data/processed/landmarks/ folder. All the scripts need to be run from the root folder.
4. Model training and evaluation scripts are being kept [here](entrega2/src/). All the scripts need to be run from the root folder.
5. To run the real time detection system, please refer to the [entrega3/src](entrega3/src/) folder. All the scripts need to be run from the root folder.


## Use of Generative AI (IAG)

Generative AI tools (ChatGPT) were used as support throughout this project for the following purposes:

- refining code structure, debugging, and identifying errors,

- generating alternative implementations during experimentation,

- clarifying theoretical concepts related to feature engineering, model evaluation, and deployment,

- improving the clarity, structure, and technical wording of documentation, including this README,

- assisting in drafting analytical sections that were later verified and completed by the author.

All modeling decisions, dataset construction, experiments, interpretations, real-time implementation, and final edits were carried out by the project author. Generative AI served strictly as an auxiliary tool and not as a substitute for academic or technical work.