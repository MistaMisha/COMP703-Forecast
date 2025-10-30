# 🧠 AI-Powered Inventory Forecasting for Optimized Stock Management  
### Shosha Sales Forecasting App – **V4 Hybrid Statistical Forecasting**

This project is a Flask-based web application that powers **AI-driven demand forecasting** across multiple retail outlets.  
It leverages **hybrid statistical and machine-learning methods** (including **LightGBM**) to generate 14-day forecasts of recommended stock quantities per outlet and per product.

---

## 🚀 Key Features

- **Hybrid Statistical Forecasting (V4):** Combines rule-based velocity segmentation, bias correction, and LightGBM-based pattern learning  
- **Multi-Outlet, Multi-Product Forecasting:** Optimized for Shosha’s nationwide retail network  
- **Recursive 14-Day Forecasts:** Adaptive prediction horizon  
- **Dynamic Web Interface:** Upload CSV files and view interactive forecast results instantly  
- **Velocity Segmentation:** Automatically classifies SKUs into High / Medium-High / Medium / Low demand tiers  
- **Duplicate & Missing Data Handling:** Automatic cleaning and aggregation  
- **Admin & Outlet User Roles:** Secure login with hashed credentials  

---

## 🧩 Tech Stack

- **Backend:** Flask (Python 3.9+)  
- **Frontend:** HTML, CSS, Jinja2 templates  
- **ML Core:** Hybrid Statistical Engine (v4)  
- **Data Layer:** pandas + NumPy  
- **Deployment:** Gunicorn (production server)  

---

## 🛠 Requirements

- Python **3.9+**  
- **pip** (Python package manager)

---

## 🧾 Steps to Run the App

```bash
# 1) Clone the repository
git clone https://github.com/Kandyli007/AI-Powered-Inventory-Forecasting-for-Optimized-Stock-Management-.git
cd AI-Powered-Inventory-Forecasting-for-Optimized-Stock-Management-

# 3) Install dependencies
pip install -r requirements.txt

# 4) Go to the app folder
cd V4-Hybrid Statistical Forecasting

# 5) Run the app
python app.py

# 6) Open in your browser
# URL:
http://127.0.0.1:5000/
```

```txt
Flask==3.0.3
Werkzeug==3.0.3
pandas==2.2.2
numpy==1.26.4
joblib==1.4.2
gunicorn==22.0.0
bcrypt==4.1.2
```
# 7) Login credentials
- username: admin
- password: pinkfrosteddonut

