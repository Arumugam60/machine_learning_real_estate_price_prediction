# 🏠 Real Estate Price Prediction Web App  

An end-to-end **Machine Learning project** that predicts real estate prices based on features such as location, BHK, square feet, and number of bathrooms.  
The project covers the complete ML lifecycle: data preprocessing, model training, evaluation, deployment using **Flask**, and a simple **HTML/CSS/JavaScript frontend** for user interaction.  

---

## 🚀 Features
- Data cleaning, preprocessing, and feature engineering  
- Outlier removal and cross-validation for robust predictions  
- Trained ML model (Linear Regression) with serialization using Pickle  
- Flask REST API to serve predictions  
- Frontend built with HTML, CSS, and JavaScript  
- Modular and deployment-ready structure  

---

## 🛠️ Tech Stack
- **Python**: pandas, numpy, scikit-learn, matplotlib, seaborn  
- **Flask**: backend REST API  
- **Frontend**: HTML, CSS, JavaScript  
- **Tools**: Git/GitHub, Postman (API testing)  

---

## 📂 Project Structure
real-estate-price-prediction/
│
├── model/
│ ├── bangalore_home_prices_model.pickle # Trained ML model
│ ├── columns.json # Data columns/features
│
├── server/
│ ├── server.py # Flask server
│
├── client/
│ ├── app.html # Frontend HTML file
│ ├── style.css # Styling
│ ├── app.js # JS for API calls
