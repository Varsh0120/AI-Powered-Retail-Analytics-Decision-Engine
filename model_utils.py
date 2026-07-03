# model_utils_safe.py
import pandas as pd
import numpy as np
from prophet import Prophet
from datetime import timedelta
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import os

#-----------------------------
#Compute_metrics [claude]
#-----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mask = y_true != 0
    if mask.sum() == 0:
        mape = None
    else:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE": round(mape, 2) if mape is not None else "N/A"}


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(filepath):
    """Load CSV and parse dates, rename columns."""
    try:
        df = pd.read_csv(filepath, parse_dates=['Date'])
        df.rename(columns={
            "Product Category": "product_category",
            "Quantity": "quantity",
            "Price per Unit": "price",
            "Total Amount": "total_amount"
        }, inplace=True)

        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df['DayOfWeek'] = df['Date'].dt.dayofweek

        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return pd.DataFrame()

# -----------------------------
# AGGREGATE CATEGORY
# -----------------------------
def aggregate_category(df):
    if df.empty:
        return pd.DataFrame()
    if 'product_category' not in df.columns or 'quantity' not in df.columns:
        return pd.DataFrame()
    df_grouped = df.groupby(['Date','product_category'])['quantity'].sum().reset_index()
    return df_grouped

# -----------------------------
# FORECASTING HELPERS
# -----------------------------
def _check_category_data(df_grouped, category):
    cat_df = df_grouped[df_grouped['product_category']==category].copy()
    if cat_df.empty:
        return None
    return cat_df

def _generate_dummy_forecast(df_grouped, category, n_days):
    last_date = pd.to_datetime(df_grouped['Date'].max()) if not df_grouped.empty else pd.Timestamp.today()
    dates = [last_date + timedelta(days=i) for i in range(1, n_days+1)]
    forecasted_quantity = np.random.randint(10,100,size=n_days)
    #-------------------------------[claude]
    return list(zip(dates, forecasted_quantity)), {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
    #return list(zip(dates, forecasted_quantity))

# -----------------------------
# PROPHET
# -----------------------------
def forecast_sales_prophet(df_grouped, category, n_days=30):
    cat_df = _check_category_data(df_grouped, category)
    if cat_df is None or len(cat_df) < 2:
        return _generate_dummy_forecast(df_grouped, category, n_days)

    cat_df = cat_df[['Date','quantity']].rename(columns={'Date':'ds','quantity':'y'})

    model = Prophet(daily_seasonality=True)
    model.fit(cat_df)

    future = model.make_future_dataframe(periods=n_days)
    forecast = model.predict(future)

#------------------------------
#forecast_sales_prophet [claude]
#------------------------------
# Evaluate on test set
    train = cat_df.iloc[:int(len(cat_df)*0.8)]
    test = cat_df.iloc[int(len(cat_df)*0.8):]
    eval_model = Prophet(daily_seasonality=True)
    eval_model.fit(train)
    test_forecast = eval_model.predict(eval_model.make_future_dataframe(periods=len(test)))
    metrics = compute_metrics(test['y'].values, test_forecast['yhat'].tail(len(test)).values)

#results = list(zip(forecast['ds'].tail(n_days), forecast['yhat'].tail(n_days)))
#return results, metrics


    results = list(zip(forecast['ds'].tail(n_days), forecast['yhat'].tail(n_days)))
    return results, metrics

# -----------------------------
# XGBOOST
# -----------------------------
def forecast_sales_xgboost(df_grouped, category, n_days=30):
    cat_df = _check_category_data(df_grouped, category)
    if cat_df is None or len(cat_df) < 2:
        return _generate_dummy_forecast(df_grouped, category, n_days)

    cat_df = cat_df.sort_values("Date")
    cat_df["dayofweek"] = cat_df["Date"].dt.dayofweek
    cat_df["month"] = cat_df["Date"].dt.month
    cat_df["year"] = cat_df["Date"].dt.year
    cat_df["dayofyear"] = cat_df["Date"].dt.dayofyear

    X = cat_df[["dayofweek","month","year","dayofyear"]]
    y = cat_df["quantity"]

    model = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X,y)

    last_date = cat_df["Date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1,n_days+1)]

    future_df = pd.DataFrame({
        "dayofweek":[d.weekday() for d in future_dates],
        "month":[d.month for d in future_dates],
        "year":[d.year for d in future_dates],
        "dayofyear":[d.timetuple().tm_yday for d in future_dates]
    })

    future_df["Forecasted Quantity"] = model.predict(future_df)

#-------------------------------
#forecast_sales_xgboost [claude]
#-------------------------------
    split = int(len(X)*0.8)
    model_eval = XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    model_eval.fit(X[:split], y[:split])
    metrics = compute_metrics(y[split:].values, model_eval.predict(X[split:]))

    results = list(zip(future_dates, future_df["Forecasted Quantity"]))
    return results, metrics

    #results = list(zip(future_dates, future_df["Forecasted Quantity"]))
    #return results

# -----------------------------
# RANDOM FOREST
# -----------------------------
def forecast_sales_random_forest(df_grouped, category, n_days=30):
    cat_df = _check_category_data(df_grouped, category)
    if cat_df is None or len(cat_df) < 2:
        return _generate_dummy_forecast(df_grouped, category, n_days)

    cat_df = cat_df.sort_values("Date")
    cat_df["dayofweek"] = cat_df["Date"].dt.dayofweek
    cat_df["month"] = cat_df["Date"].dt.month
    cat_df["year"] = cat_df["Date"].dt.year
    cat_df["dayofyear"] = cat_df["Date"].dt.dayofyear

    X = cat_df[["dayofweek","month","year","dayofyear"]]
    y = cat_df["quantity"]

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X,y)

    last_date = cat_df["Date"].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1,n_days+1)]

    future_df = pd.DataFrame({
        "dayofweek":[d.weekday() for d in future_dates],
        "month":[d.month for d in future_dates],
        "year":[d.year for d in future_dates],
        "dayofyear":[d.timetuple().tm_yday for d in future_dates]
    })

    future_df["Forecasted Quantity"] = model.predict(future_df)
    
    #-------------------------------
    #forecast_sales_random_forest 
    #-------------------------------
    split = int(len(X)*0.8)
    model_eval = RandomForestRegressor(n_estimators=200, random_state=42)
    model_eval.fit(X[:split], y[:split])
    metrics = compute_metrics(y[split:].values, model_eval.predict(X[split:]))

    results = list(zip(future_dates, future_df["Forecasted Quantity"]))
    return results, metrics


    #results = list(zip(future_dates, future_df["Forecasted Quantity"]))
    #return results

# -----------------------------
# LSTM
# -----------------------------
def forecast_sales_lstm(df_grouped, category, n_days=30):
    cat_df = _check_category_data(df_grouped, category)
    if cat_df is None or len(cat_df) < 10:
        return _generate_dummy_forecast(df_grouped, category, n_days)

    data = cat_df["quantity"].values.reshape(-1,1)
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    SEQ_LEN = 10
    X=[]
    y=[]
    for i in range(10,len(data_scaled)):
        X.append(data_scaled[i-10:i])
        y.append(data_scaled[i])

    X=np.array(X)
    y=np.array(y)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    #model = Sequential()
    #model.add(LSTM(50,return_sequences=True,input_shape=(X.shape[1],1)))
    #model.add(LSTM(50))
    #model.add(Dense(1))
    #model.compile(optimizer='adam',loss='mse')
#----------------------------
#LSTM Dropout [claude]
#------------------------------

    from tensorflow.keras.layers import Dropout

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse') 

    model.fit(X_train,y_train,epochs=30,batch_size=16,validation_split=0.1,verbose=0)

    last_sequence = data_scaled[-10:]
    current_seq = last_sequence
    predictions=[]

    for i in range(n_days):
        pred = model.predict(current_seq.reshape(1,10,1),verbose=0)
        predictions.append(pred[0][0])
        current_seq = np.append(current_seq[1:],pred)

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1,1))
    future_dates = [cat_df["Date"].max()+timedelta(days=i) for i in range(1,n_days+1)]
    
    #----------------------------
    #forecast_sales_lstm [claude]
    #------------------------------
    y_pred_test = scaler.inverse_transform(model.predict(X_test, verbose=0)).flatten()
    y_true_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    metrics = compute_metrics(y_true_test, y_pred_test)
    results = list(zip(future_dates, predictions.flatten()))
    return results, metrics


    #results = list(zip(future_dates, predictions.flatten()))
    #return results

# -----------------------------
# STRATEGIC GUIDANCE
# -----------------------------
def generate_strategic_guidance(forecasts, base_price: float):
    guidance = []
    for date, qty in forecasts:
        guidance.append({
            "Date": date.strftime("%Y-%m-%d"),
            "Forecasted Quantity": qty,
            "Price Recommendation": "Increase" if qty < 50 else "Maintain",
            "Promotion Recommendation": "Run Promotion" if qty < 50 else "Standard Marketing"
        })
    return guidance

def summarize_recommendations(guidance: list) -> str:
    if not guidance:
        return "No recommendations available."
    total_qty = sum(item['Forecasted Quantity'] for item in guidance)
    avg_qty = total_qty / len(guidance)
    price_actions = set(item['Price Recommendation'] for item in guidance)
    promo_actions = set(item['Promotion Recommendation'] for item in guidance)
    summary = f"Total forecasted quantity: {total_qty}\n"
    summary += f"Average daily forecast: {avg_qty:.2f}\n"
    summary += f"Price strategy: {', '.join(price_actions)}\n"
    summary += f"Promotion strategy: {', '.join(promo_actions)}\n"
    return summary

# -----------------------------
# EXPORT RECOMMENDATIONS
# -----------------------------
def export_recommendations_csv(guidance: list, folder_path: str = "./exports") -> str:
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    df = pd.DataFrame(guidance)
    file_path = os.path.join(folder_path, f"strategic_guidance_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
    df.to_csv(file_path, index=False)
    return file_path

# -----------------------------
# MODEL COMPARISON
# -----------------------------
def compare_all_models(df_grouped, category, n_days=30):
    """Runs all 4 models and returns their metrics for comparison."""
    results = {}
    
    _, prophet_metrics = forecast_sales_prophet(df_grouped, category, n_days)
    results["Prophet"] = prophet_metrics
    
    _, xgb_metrics = forecast_sales_xgboost(df_grouped, category, n_days)
    results["XGBoost"] = xgb_metrics
    
    _, rf_metrics = forecast_sales_random_forest(df_grouped, category, n_days)
    results["Random Forest"] = rf_metrics
    
    _, lstm_metrics = forecast_sales_lstm(df_grouped, category, n_days)
    results["LSTM"] = lstm_metrics
    
    return results

#Select the best model based on RMSE #NEW
def select_best_model(results):
    valid_models = {
        model: metrics["RMSE"]
        for model, metrics in results.items()
        if metrics["RMSE"] != "N/A"
    }

    if not valid_models:
        return None, None

    best_model = min(valid_models, key=valid_models.get)
    return best_model, valid_models[best_model]