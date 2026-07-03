import streamlit as st
import pandas as pd
import plotly.express as px
from io import StringIO

from model_utils import (
    load_data,
    aggregate_category,
    forecast_sales_prophet,
    forecast_sales_xgboost,
    forecast_sales_random_forest,
    forecast_sales_lstm,
    generate_strategic_guidance,
    summarize_recommendations,
    export_recommendations_csv,
    compare_all_models,
    select_best_model  
)

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Business Intelligence and Decision Engine",
    layout="wide"
)

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown("""
<style>
html, body {
    font-family: 'Inter', sans-serif;
    background-color: #f5f8fb;
}
.main-title {
    background: linear-gradient(90deg, #007c91, #00a8b5);
    color: white;
    padding: 1.2rem;
    border-radius: 12px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 600;
    margin-bottom: 25px;
}
.metric-card {
    background-color: white;
    border-radius: 15px;
    padding: 1.2rem;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #007c91;
}
.metric-label {
    color: #666;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    "<div class='main-title'>🤖 AI Business Intelligence and Decision Engine</div>",
    unsafe_allow_html=True
)

# -----------------------------
# LOAD DATA [claude]
# -----------------------------
st.sidebar.header("Data Upload")
uploaded_file = st.sidebar.file_uploader("Upload Retail Sales CSV", type=["csv"])

if uploaded_file is not None:
    df = load_data(uploaded_file)
else:
    st.info("👈 Please upload your RetailSalesDataset CSV file in the sidebar to begin.")
    st.stop()

if df.empty:
    st.error("Could not load data. Please check your CSV format.")
    st.stop()

df_grouped = aggregate_category(df)

# ----------------------------
# DATA EXPLORATION TAB
# ----------------------------
st.subheader("📊 Data Exploration")
with st.expander("Click to explore raw data"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{len(df)}</div>
            <div class='metric-label'>Total Records</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{df['product_category'].nunique()}</div>
            <div class='metric-label'>Categories</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class='metric-card'>
            <div class='metric-value'>{df['Date'].min().strftime('%Y-%m')}</div>
            <div class='metric-label'>Data From</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Missing values
    st.write("**Missing Values:**")
    missing = df.isnull().sum()
    st.dataframe(missing[missing > 0] if missing.sum() > 0 else pd.DataFrame({"Status": ["No missing values"]}))

    # Sales distribution
    st.write("**Sales Distribution by Category:**")
    fig_exp = px.box(df, x="product_category", y="quantity", title="Quantity Distribution per Category")
    st.plotly_chart(fig_exp, use_container_width=True)

    # Sales over time
    st.write("**Overall Sales Trend:**")
    daily_sales = df.groupby("Date")["quantity"].sum().reset_index()
    fig_trend = px.line(daily_sales, x="Date", y="quantity", title="Daily Sales Over Time")
    st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("User Inputs")

category = st.sidebar.selectbox(
    "Select Product Category",
    df_grouped['product_category'].unique(),
    key="category_select"
)

n_days = st.sidebar.slider("Forecast Days", 1, 90, 30)

base_price = st.sidebar.number_input("Base Price per Unit", min_value=1, value=100)

model_choice = st.sidebar.radio(
    "Select Forecasting Model",
    ["Prophet", "XGBoost", "Random Forest", "LSTM"]
)

# -----------------------------
# MODEL COMPARISON [claude]
# -----------------------------
if st.sidebar.button("🔍 Compare All Models"):
    with st.spinner("Running all 4 models... this may take a minute"):
        comparison = compare_all_models(df_grouped, category, n_days)
        best_model, best_rmse = select_best_model(comparison)  #NEW

    st.subheader("📊 Model Comparison")
    st.caption("All models evaluated on last 20% of historical data")
    
    if best_model: #NEW
        st.success(f"🏆 Best Model: {best_model} | Lowest RMSE: {best_rmse:.2f}")

    comp_df = pd.DataFrame(comparison).T.reset_index()
    comp_df.columns = ["Model", "MAE", "RMSE", "MAPE"]
    st.dataframe(comp_df)

    fig_mae = px.bar(comp_df, x="Model", y="MAE",
                     title="MAE Comparison (lower is better)", color="Model")
    st.plotly_chart(fig_mae, use_container_width=True)

    fig_rmse = px.bar(comp_df, x="Model", y="RMSE",
                      title="RMSE Comparison (lower is better)", color="Model")
    st.plotly_chart(fig_rmse, use_container_width=True)

    st.markdown("---")

# -----------------------------
# GENERATE FORECAST
# -----------------------------
if st.sidebar.button("Generate Forecast", key="gen_forecast"):

    # --------------------------------
    # WITH METRIC [claude]
    # --------------------------------
    if model_choice == "Prophet":
        forecasts, metrics = forecast_sales_prophet(df_grouped, category, n_days)
    elif model_choice == "XGBoost":
        forecasts, metrics = forecast_sales_xgboost(df_grouped, category, n_days)
    elif model_choice == "Random Forest":
        forecasts, metrics = forecast_sales_random_forest(df_grouped, category, n_days)
    else:
        forecasts, metrics = forecast_sales_lstm(df_grouped, category, n_days)

    if forecasts is not None and len(forecasts) > 0:

        forecast_df = pd.DataFrame(forecasts, columns=["Date", "Forecasted Quantity"])
        forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
        forecast_df["Week"] = forecast_df["Date"].dt.isocalendar().week
        forecast_df["Category"] = category
        forecast_df["MonthStr"] = forecast_df["Date"].dt.strftime("%Y-%m")

        # -----------------------------
        # KPI CALCULATIONS
        # -----------------------------
        total_sales = int(forecast_df["Forecasted Quantity"].sum())

        #NEW
        forecast_df["Forecast Revenue"] = (
            forecast_df["Forecasted Quantity"] * base_price
        )

        total_revenue = forecast_df["Forecast Revenue"].sum()
        avg_revenue = forecast_df["Forecast Revenue"].mean() #NEW

        avg_sales = round(forecast_df["Forecasted Quantity"].mean(), 2)
        price_strategy = "Maintain" if avg_sales > 50 else "Discount"
        promo_strategy = "Run Promotion" if avg_sales < 50 else "Standard Marketing"

        # NEW
        if avg_sales >= 80:
            inventory_status = "🟢 High Inventory Required"
        elif avg_sales >= 40:
            inventory_status = "🟡 Medium Inventory Required"
        else:
            inventory_status = "🔴 Low Inventory Required" #NEW

        # -----------------------------
        # KPI DISPLAY
        # -----------------------------
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{total_sales}</div><div class='metric-label'>Total Forecasted Sales</div></div>", unsafe_allow_html=True)
        with k2:
            #NEW
            st.markdown(
        f"<div class='metric-card'><div class='metric-value'>₹{total_revenue:,.0f}</div><div class='metric-label'>Forecast Revenue</div></div>",
        unsafe_allow_html=True
        ) #NEW
        with k3:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{price_strategy}</div><div class='metric-label'>Pricing Strategy</div></div>", unsafe_allow_html=True)
        with k4:
            st.markdown(f"<div class='metric-card'><div class='metric-value'>{promo_strategy}</div><div class='metric-label'>Promotion Strategy</div></div>", unsafe_allow_html=True)

        st.markdown("---")

        # -----------------------------
        # MODEL EVALUATION METRICS 
        # -----------------------------
        st.subheader("📐 Model Evaluation Metrics")
        st.caption("Based on last 20% of historical data — not future forecasts")
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(
                "MAE",
                metrics.get("MAE", "N/A"),
                help="Average units the forecast was wrong by"
            )

        with m2:
            st.metric(
                "RMSE",
                metrics.get("RMSE", "N/A"),
                help="Penalises large errors more heavily"
            )

        with m3:
            mape = metrics.get("MAPE", "N/A")
            st.metric(
                "MAPE",
                f"{mape}%" if isinstance(mape, (int, float)) else mape,
                help="Percentage error — lower is better"
            )

# NEW
        if metrics.get("RMSE") != "N/A":
            if metrics["RMSE"] < 10:
                st.success("✅ Excellent forecast accuracy")
            elif metrics["RMSE"] < 20:
                st.info("ℹ️ Good forecast accuracy")
            else:
                st.warning("⚠️ Forecast accuracy could be improved")

        st.markdown("---") #NEW

        # -----------------------------
        # FORECAST GRAPH
        # -----------------------------
        st.subheader("Sales Forecast Trend")
        fig = px.area(forecast_df, x="Date", y="Forecasted Quantity", title=f"Forecast for {category}")
        st.plotly_chart(fig, use_container_width=True)

        # -----------------------------
        # WEEKLY FORECAST
        # -----------------------------
        st.subheader("Weekly Forecast")
        weekly_df = forecast_df.groupby("Week")["Forecasted Quantity"].sum().reset_index()
        fig2 = px.bar(weekly_df, x="Week", y="Forecasted Quantity")
        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
        # CUMULATIVE SALES
        # -----------------------------
        forecast_df["Cumulative Sales"] = forecast_df["Forecasted Quantity"].cumsum()
        st.subheader("Cumulative Sales Forecast")
        fig3 = px.line(forecast_df, x="Date", y="Cumulative Sales")
        st.plotly_chart(fig3, use_container_width=True)

        # -----------------------------
        # MONTHLY TREND
        # -----------------------------
        st.subheader("Monthly Forecast Trend")
        monthly_df = forecast_df.groupby("MonthStr")["Forecasted Quantity"].sum().reset_index()
        fig4 = px.line(monthly_df, x="MonthStr", y="Forecasted Quantity", markers=True)
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("---")

        # -----------------------------
        # FEATURE IMPORTANCE [claude]
        # -----------------------------
        if model_choice in ["XGBoost", "Random Forest"]:
            st.subheader("🔍 Feature Importance")
            st.caption("Shows which time features most influenced the forecast")
            from xgboost import XGBRegressor
            from sklearn.ensemble import RandomForestRegressor
            cat_df = df_grouped[df_grouped['product_category'] == category].copy()
            cat_df["dayofweek"] = cat_df["Date"].dt.dayofweek
            cat_df["month"] = cat_df["Date"].dt.month
            cat_df["year"] = cat_df["Date"].dt.year
            cat_df["dayofyear"] = cat_df["Date"].dt.dayofyear
            X = cat_df[["dayofweek", "month", "year", "dayofyear"]]
            y = cat_df["quantity"]
            if model_choice == "XGBoost":
                fi_model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5)
            else:
                fi_model = RandomForestRegressor(n_estimators=200)
            fi_model.fit(X, y)
            importances = fi_model.feature_importances_
            fi_df = pd.DataFrame({
                "Feature": ["Day of Week", "Month", "Year", "Day of Year"],
                "Importance": importances
            }).sort_values("Importance", ascending=False)
            fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                            title=f"Feature Importance — {model_choice}",
                            color="Importance", color_continuous_scale="teal")
            st.plotly_chart(fig_fi, use_container_width=True)

        # -----------------------------
        # STRATEGIC GUIDANCE
        # -----------------------------
        
        #NEW
        st.subheader("📦 Inventory Recommendation")

        st.success(inventory_status)

        st.markdown("---") #NEW

        strategic_guidance = generate_strategic_guidance(forecasts, base_price)
        guidance_df = pd.DataFrame(strategic_guidance)
        st.dataframe(guidance_df)
        csv_path = export_recommendations_csv(strategic_guidance)
        st.success(f"Strategic guidance exported: {csv_path}")

        # -----------------------------
        # AI RECOMMENDATION
        # -----------------------------
        st.subheader("AI Recommendation")
        if avg_sales > 60:
            insight = f"Sales for {category} are expected to grow. Maintain inventory and promote bundles."
        else:
            insight = f"Demand for {category} is moderate. Consider discounts or targeted promotions."
        st.info(insight)

        # -----------------------------
        # SUMMARY
        # -----------------------------
        st.subheader("Recommendation Summary")
        summary_text = summarize_recommendations(strategic_guidance)
        st.text_area("Summary", summary_text, height=200)
        buffer = StringIO()
        buffer.write(summary_text)
        st.download_button(
            label="Download Summary",
            data=buffer.getvalue(),
            file_name=f"{category}_summary.txt",
            mime="text/plain"
        )

    else:
        st.error("Forecast could not be generated.")


#python --version
#pip --version
#python -m pip show streamlit
        #python -m pip install tensorflow
        #pip install streamlit pandas numpy scikit-learn xgboost prophet tensorflow plotly
        #streamlit run app.py or python -m streamlit run app.py
        #python -m venv .venv
        #.venv\Scripts\activate