# Hotel Booking Cancellation – Predictive Modeling

This project analyzes hotel reservation data to predict the probability of a booking being canceled. Using real-world data from over 36,000 reservations, various machine learning models were trained and compared to estimate the impact of different features on cancellation behavior.

## 📊 Dataset
- Source: Kaggle (Ashan Raza)
- 36,275 hotel reservations, 9 key features
- Mixed numerical and categorical variables
- Target: `booking_status` (Canceled / Not Canceled)

## 🎯 Objective
To develop a predictive model capable of estimating the cancellation probability of a hotel reservation using pre-booking information and customer behavior patterns.

## 🧠 Models Used
- Linear Regression
- Ridge & Lasso Regression
- Decision Trees
- Random Forests
- K-Nearest Neighbors (KNN)

## 🛠️ Technologies
- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib, Seaborn

## 📈 Evaluation Metrics
- Root Mean Squared Error (RMSE)
- Mean Squared Error (MSE)
- R² Score

## 📝 Results Summary
- Best-performing model: **KNN with k=10**
  - RMSE: **27.75**
- Models compared using train/test split
- Interpretation of key variables provided

## 📂 Files
- `hotel_analysis.ipynb`: Main notebook with preprocessing, modeling, and evaluation
- `report.pdf`: Full written report with context, visuals and interpretation
- `/figures`: Visualizations and correlation plots

## 🔗 Author
Víctor Suesta  
[LinkedIn](https://www.linkedin.com/in/víctor-suesta-arribas-7b1250322/)
