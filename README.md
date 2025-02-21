**Disaster Response (Udacity Project 2)**
<br><br>**Table of Contents**
<br>Summary
<br>File Structure
<br>How to Run the Python Scripts and Web App
<br>Prerequisites
<br>Installation Steps
<br>Running the Data Processing Script
<br>Running the Flask Web App
<br>
<br>
<br>**Summary**
<br>This GitHub repository the files required for the Disaster Response Project 2 for the Udacity Data Science nano degree. The goal of the project was to create a Natural Language Processing model to classify disaster-related messaged from social media, and classify them into multiple categories. 
<br>The project includes:
<br>ETL Process: Python scripts load, clean, and store disaster messages and categories in a SQLite database.
<br>Machine Learning Model: A trained natural language processing model that classifies incoming messages into categories. (saved as a pickle file)
<br>Web Application: A Flask-based web app to visualise the data and allow users to classify new messages.
<br>
<br>
<br>**File Structure**
<br>README: - Current file.
<br>
<br>app folder:
<br>templates folder:
<br>go.html - html file for Flask app.
<br>master.html - html file for Flask app.
<br>run.py - Script to run Flask app which displays visualisations and classify disaster response messages.
<br><br>data folder:
<br>process_data.py - Script to load, clean and save data to a SQLite database.
<br>categories.csv - Original categories dataset in CSV format.
<br>messages.csv - Original messages dataset in CSV format.
<br>data.db - Cleaned and merged data in SQLite database.
<br><br>models folder:
<br>train_classifier.py - Script to load data, train model and save to pickle file.
<br>trained_model.pkl - Trained machine learning model for classifying disaster response messages into categories. (File too large, even using LFS, for GitHub so not present, advised by Udacity mentor it’s fine to submit without the file)
<br>
<br>**How to Run the Python Scripts and Web App**
<br>
<br>**Prerequisites**
<br>Python 3.x
<br>Git
<br>SQLite (for data storage)
<br>Recommended: Use a virtual environment
<br><br>
<br>**Installation Steps**
<br>Clone the Repository: ```bash  git clone https://github.com/SusieCrone/DataScienceProject2.git
<br>cd DataScienceProject2
<br><br> 
<br>**Running the Data Processing Script**
<br>The process_data.py script loads, cleans, and stores the disaster messages and categories into a SQLite database. Run it using:
<br>bash
<br>python process_data.py messages.csv categories.csv data.db
<br>This command will:
<br>Load the messages and categories datasets.
<br>Clean and merge the data.
<br>Save the cleaned data to a SQLite database (data.db)
<br><br> 
<br>**Running the Flask Web App**
<br>The  process launched the app which displays interactive Plotly graphs and provides an interface for classifying disaster messages using the trained model.
<br>bash
<br>python run.py
<br><br>
<br>Then open your web browser and go to:
<br>http://127.0.0.1:3000


