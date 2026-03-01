# Sentiment Analysis Web App

A Flask web application that analyzes the sentiment of Amazon product reviews (sourced from Kaggle's [Amazon Musical Instrument Reviews](https://www.kaggle.com/datasets/eswarchandt/amazon-music-reviews) dataset) using machine learning. The app classifies reviews as **Positive**, **Negative**, or **Neutral**.

## Features

- Real-time sentiment analysis of product reviews
- Clean web interface
- Machine learning model trained on Amazon musical instruments reviews
- Supports text preprocessing and TF-IDF vectorization

## Prerequisites

- Python 3.7+
- pip (Python package manager)

## Setup

```bash
# Create virtual environment
python -m venv sentiment_env

# Activate virtual environment

# On Windows:
sentiment_env\Scripts\activate

# On macOS/Linux:
source sentiment_env/bin/activate

## Running the Application

- Make sure your virtual environment is activated

- Run the Flask application:

``` 
python final_app.py

Open your web browser and go to:

http://localhost:5000