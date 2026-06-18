from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import joblib
import os
import json
import pandas as pd

# Import your RAG AI function
from ai_agent import process_query

app = FastAPI()

# Allow the frontend to communicate with this backend API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------
# Load the trained Machine Learning model
# ---------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'sales_model.pkl')
try:
    sales_model = joblib.load(MODEL_PATH)
    print("✅ ML Model loaded successfully.")
except Exception as e:
    sales_model = None
    print(f"⚠️ Warning: ML model not found at {MODEL_PATH}. Please run train_model.py first.")

# ---------------------------------------------------------
# UI ROUTE
# ---------------------------------------------------------
@app.get("/")
@app.get("/dashboard")
async def serve_dashboard():
    """Serves the main HTML dashboard"""
    html_path = os.path.join(os.path.dirname(__file__), 'dashboard.html')
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"status": "error", "message": f"dashboard.html file not found at {html_path}"}

# ---------------------------------------------------------
# EXISTING API ROUTES
# ---------------------------------------------------------
@app.get("/api/ask")
async def ask_ai_agent(query: str):
    """Handles Retrieval-Augmented Generation (RAG) Q&A for text documents"""
    try:
        ai_response = process_query(query)
        return {"status": "success", "user_query": query, "ai_response": ai_response}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/revenue")
async def get_live_revenue():
    """Fetches the real-time data written by PySpark"""
    try:
        json_path = os.path.join(os.path.dirname(__file__), 'live_sales.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)
            return {"status": "success", "data": data}
        else:
            return {"status": "success", "data": []}
    except Exception as e:
         return {"status": "error", "message": str(e)}

@app.get("/api/predict")
async def predict_revenue(quantity: int, discount: float, region: int):
    """Predicts future revenue using the Random Forest ML model"""
    if not sales_model:
        return {"status": "error", "message": "ML model not found. Run train_model.py first."}
    try:
        input_features = [[quantity, discount, region]]
        prediction = sales_model.predict(input_features)[0]
        return {"status": "success", "predicted_revenue": round(prediction, 2)}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/charts")
async def get_chart_data():
    """Aggregates historical data for the frontend visualization graphs"""
    try:
        data_path = os.path.join(os.path.dirname(__file__), '../data/superstore.csv')
        df = pd.read_csv(data_path, encoding='windows-1252')
        
        region_sales = df.groupby('Region')['Sales'].sum().round(2).to_dict()
        category_sales = df.groupby('Category')['Sales'].sum().round(2).to_dict()
        region_discount = df.groupby('Region')['Discount'].mean().round(3).to_dict()

        return {
            "status": "success",
            "region_sales": region_sales,
            "category_sales": category_sales,
            "region_discount": region_discount
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/report")
async def generate_pdf_report():
    """Generates a professional PDF report with embedded Matplotlib graphs and AI predictions"""
    from fpdf import FPDF
    import datetime
    import matplotlib
    import matplotlib.pyplot as plt
    import uuid

    # Force Matplotlib to run silently in the background
    matplotlib.use('Agg')

    try:
        # --- 1. Gather Historical & Live Data ---
        data_path = os.path.join(os.path.dirname(__file__), '../data/superstore.csv')
        df = pd.read_csv(data_path, encoding='windows-1252')

        json_path = os.path.join(os.path.dirname(__file__), 'live_sales.json')
        live_data = []
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                live_data = json.load(f)

        total_revenue = sum(item.get("Total_Revenue", 0) for item in live_data)
        top_region = max(live_data, key=lambda x: x.get("Total_Revenue", 0)).get("Region") if live_data else "N/A"

        # --- 2. Calculate AI Time-Series Forecast ---
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        df['Order Date'] = pd.to_datetime(df['Order Date'])
        monthly_sales = df.resample('ME', on='Order Date')['Sales'].sum()
        model = ExponentialSmoothing(monthly_sales, trend='add', seasonal='add', seasonal_periods=12)
        fit_model = model.fit()
        forecast = fit_model.forecast(6)
        next_month_pred = forecast.iloc[0]

        # --- 3. Generate High-Res Graphs for the PDF ---
        # We use UUID to prevent file name collisions if multiple users generate reports at once
        chart_filename_1 = f"temp_pie_{uuid.uuid4().hex}.png"
        chart_filename_2 = f"temp_line_{uuid.uuid4().hex}.png"

        # Graph A: Regional Distribution
        region_sales = df.groupby('Region')['Sales'].sum()
        plt.figure(figsize=(6, 4))
        plt.pie(region_sales, labels=region_sales.index, autopct='%1.1f%%', colors=['#0ea5e9', '#10b981', '#f59e0b', '#8b5cf6'])
        plt.title('Historical Sales Distribution by Region')
        plt.savefig(chart_filename_1, bbox_inches='tight')
        plt.close()

        # Graph B: Forecasting Trend Line
        plt.figure(figsize=(8, 4))
        plt.plot(monthly_sales.index[-24:], monthly_sales.values[-24:], label='Historical (Last 2 Yrs)', color='#0ea5e9', marker='o')
        plt.plot(forecast.index, forecast.values, label='AI Forecast (Next 6 Mo)', color='#f59e0b', linestyle='--', marker='o')
        plt.title('AI Revenue Forecast (Holt-Winters)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.savefig(chart_filename_2, bbox_inches='tight')
        plt.close()

        # --- 4. Build the Professional PDF ---
        pdf = FPDF()
        pdf.add_page()
        
        # Header
        pdf.set_font("Arial", 'B', 18)
        pdf.set_text_color(15, 23, 42)
        pdf.cell(0, 10, txt="Enterprise Intelligence Suite", ln=True, align='C')
        pdf.set_font("Arial", 'B', 14)
        pdf.set_text_color(56, 189, 248)
        pdf.cell(0, 10, txt="Comprehensive AI & Analytics Report", ln=True, align='C')
        pdf.ln(5)

        # Section 1: Narrative
        date_str = datetime.datetime.now().strftime("%B %d, %Y")
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, txt="1. Executive Summary & Live Operations", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(71, 85, 105)
        
        narrative = (f"Report generated on {date_str}. The real-time streaming infrastructure is currently tracking "
                     f"a live session revenue of ${total_revenue:,.2f}. The historical dataset highlights {top_region} "
                     f"as the primary revenue driver.")
        pdf.multi_cell(0, 6, txt=narrative)
        pdf.ln(5)

        # Section 2: Visual Market Share
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, txt="2. Visual Analytics & Market Share", ln=True)
        pdf.image(chart_filename_1, x=50, w=110)
        pdf.ln(5)

        # Section 3: Time-Series Predictions
        pdf.cell(0, 8, txt="3. AI Time-Series Forecasting", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(0, 6, txt=f"The Holt-Winters exponential smoothing model projects a revenue of ${next_month_pred:,.2f} for the upcoming month.")
        pdf.image(chart_filename_2, x=20, w=170)
        pdf.ln(5)

        # Section 4: Random Forest Baseline Predictor
        pdf.set_font("Arial", 'B', 12)
        pdf.set_text_color(0, 0, 0)
        pdf.cell(0, 8, txt="4. Random Forest Baseline Predictions", ln=True)
        pdf.set_font("Arial", '', 11)
        pdf.set_text_color(71, 85, 105)
        
        # Access the global ML model to make a live baseline prediction
        global sales_model
        rf_text = "Machine Learning model is not currently loaded to provide baseline scenarios."
        if sales_model:
            try:
                # Standard hypothetical baseline: Quantity 5, Discount 10%, West Region (1)
                pred_scenario = sales_model.predict([[5, 0.1, 1]])[0]
                rf_text = (f"Based on our trained Random Forest Regressor, a standard future transaction "
                           f"(5 items, 10% discount, West region) is predicted to generate ${pred_scenario:,.2f} in revenue. "
                           "The model evaluates item quantity and discount rate as the highest contributing features to variance.")
            except Exception as e:
                pass
        pdf.multi_cell(0, 6, txt=rf_text)

        # --- 5. Clean up temporary chart images ---
        if os.path.exists(chart_filename_1): os.remove(chart_filename_1)
        if os.path.exists(chart_filename_2): os.remove(chart_filename_2)

        # Save and trigger download
        report_path = os.path.join(os.path.dirname(__file__), "Comprehensive_AI_Report.pdf")
        pdf.output(report_path)

        return FileResponse(
            report_path, 
            media_type='application/pdf', 
            filename="Comprehensive_AI_Report.pdf"
        )
        
    except Exception as e:
        # Emergency cleanup just in case the code crashes midway
        for f in [chart_filename_1, chart_filename_2]:
            if 'f' in locals() and os.path.exists(f): os.remove(f)
        return {"status": "error", "message": str(e)}

# ---------------------------------------------------------
# NEW ADVANCED ML ROUTES
# ---------------------------------------------------------
@app.get("/api/forecast")
async def get_time_series_forecast():
    """Generates a 6-month AI Time-Series Forecast using Holt-Winters Exponential Smoothing"""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        import numpy as np

        data_path = os.path.join(os.path.dirname(__file__), '../data/superstore.csv')
        df = pd.read_csv(data_path, encoding='windows-1252')

        df['Order Date'] = pd.to_datetime(df['Order Date'])
        monthly_sales = df.resample('ME', on='Order Date')['Sales'].sum()

        model = ExponentialSmoothing(monthly_sales, trend='add', seasonal='add', seasonal_periods=12)
        fit_model = model.fit()
        forecast = fit_model.forecast(6)

        labels = [d.strftime('%b %Y') for d in monthly_sales.index.append(forecast.index)]
        historical_data = [round(val, 2) for val in monthly_sales.values] + [None] * 6
        forecast_data = [None] * (len(monthly_sales) - 1) + [round(monthly_sales.iloc[-1], 2)] + [round(val, 2) for val in forecast.values]

        return {
            "status": "success",
            "labels": labels,
            "historical": historical_data,
            "forecast": forecast_data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/segmentation")
async def get_customer_segmentation():
    """Performs RFM analysis and K-Means clustering for Unsupervised Customer Segmentation"""
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler

        data_path = os.path.join(os.path.dirname(__file__), '../data/superstore.csv')
        df = pd.read_csv(data_path, encoding='windows-1252')

        df['Order Date'] = pd.to_datetime(df['Order Date'])
        snapshot_date = df['Order Date'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('Customer Name').agg({
            'Order Date': lambda x: (snapshot_date - x.max()).days,
            'Order ID': 'nunique',
            'Sales': 'sum'
        }).reset_index()
        
        rfm.rename(columns={'Order Date': 'Recency', 'Order ID': 'Frequency', 'Sales': 'Monetary'}, inplace=True)

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

        cluster_data = {0: [], 1: [], 2: []}
        for _, row in rfm.iterrows():
            cluster_id = int(row['Cluster'])
            cluster_data[cluster_id].append({
                "x": int(row['Recency']),
                "y": round(row['Monetary'], 2),
                "r": min(max(int(row['Frequency']) * 1.5, 4), 25), 
                "name": row['Customer Name']
            })

        return {"status": "success", "clusters": cluster_data}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/api/diagnostics")
async def get_model_diagnostics():
    """Calculates Random Forest metrics and Feature Importance (Explainable AI)"""
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np

        data_path = os.path.join(os.path.dirname(__file__), '../data/superstore.csv')
        df = pd.read_csv(data_path, encoding='windows-1252')

        region_map = {'East': 0, 'West': 1, 'Central': 2, 'South': 3}
        df['Region_Num'] = df['Region'].map(region_map).fillna(0)
        
        X = df[['Quantity', 'Discount', 'Region_Num']]
        y = df['Sales']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)

        predictions = rf_model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        importances = rf_model.feature_importances_
        
        return {
            "status": "success",
            "metrics": {
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2": round(r2, 4)
            },
            "feature_importance": {
                "Quantity": round(importances[0] * 100, 2),
                "Discount": round(importances[1] * 100, 2),
                "Region": round(importances[2] * 100, 2)
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting API Server on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)