# app.py
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import streamlit as st
import plotly.graph_objs as go
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error
import itertools

st.set_page_config(page_title="Прогноз NVDA", layout="wide")

# Загрузка данных
@st.cache_data(show_spinner=True)
def load_data(ticker="NVDA", data_start="2020-01-01", data_end="2025-11-01"):
    data = yf.download(ticker, start=data_start, end=data_end, progress=False, auto_adjust=True)
    if data.empty:
        print("Данные не загружены.")
        return pd.DataFrame()
    new_cols = []
    for c in data.columns:
        if isinstance(c, tuple):
            new_cols.append(c[0].lower())
        else:
            new_cols.append(str(c).lower())
    data.columns = new_cols
    data.reset_index(inplace=True)
    if 'date' not in data.columns and 'Date' in data.columns:
        data.rename(columns={'Date': 'date'}, inplace=True)

    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        else:
            data[col] = pd.NA
    data['date'] = pd.to_datetime(data['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    print(f"\nРазмер данных: {data.shape[0]} строк × {data.shape[1]} столбцов.")
    print(data.head())
    print(data.info())
    print(data.describe())

    return data

# Создание БД
def create_db(df, db_name="nvidia_stock.db", table_name="nvidia"):
    if df is None or df.empty:
        return None
    db_path = os.path.join(os.path.dirname(__file__), db_name)
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                date TEXT PRIMARY KEY,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume INTEGER
            )
        """)
        conn.commit()
        df_to_insert = df[['date','open','high','low','close','volume']].copy()
        df_to_insert.to_sql(table_name, conn, if_exists='replace', index=False)
        return conn
    except Exception as e:
        st.error(f"Ошибка при создании БД: {e}")
        return None

# SQL-запросы к БД
def query_db(conn, query_name):
    queries = {
        "annual_volume": """
            SELECT STRFTIME('%Y', date) AS Year, SUM(volume) AS TotalVolume
            FROM nvidia
            GROUP BY Year
            ORDER BY Year;
        """,
        "monthly_avg_close_2025": """
            SELECT STRFTIME('%Y-%m', date) AS Month, ROUND(AVG(close),2) AS AvgClose
            FROM nvidia
            WHERE date BETWEEN '2025-01-01' AND '2025-11-30'
            GROUP BY Month
            ORDER BY Month;
        """,
        "monthly_volume_2025": """
            SELECT STRFTIME('%Y-%m', date) AS Month, SUM(volume) AS TotalVolume
            FROM nvidia
            WHERE date BETWEEN '2025-01-01' AND '2025-11-30'
            GROUP BY Month
            ORDER BY Month;
        """,
        "monthly_return_2025": """
            WITH monthly_close AS (
                SELECT STRFTIME('%Y-%m', date) AS Month, AVG(close) AS AvgClose
                FROM nvidia
                WHERE date BETWEEN '2025-01-01' AND '2025-11-30'
                GROUP BY Month
            ),
            monthly_with_lag AS (
                SELECT Month, AvgClose, LAG(AvgClose) OVER (ORDER BY Month) AS PrevAvgClose
                FROM monthly_close
            )
            SELECT Month, ROUND(((AvgClose - PrevAvgClose)/PrevAvgClose)*100,2) AS MonthlyReturnPct
            FROM monthly_with_lag
            WHERE PrevAvgClose IS NOT NULL
            ORDER BY Month;
        """
    }
    if query_name not in queries:
        st.error(f"Запрос {query_name} не найден")
        return pd.DataFrame()
    return pd.read_sql_query(queries[query_name], conn)

# Создание модели ARIMA
def train_arima_model(df, forecast_steps=10):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    series = df['close'].astype(float).interpolate()
    train_size = int(len(series)*0.8)
    train, test = series[:train_size], series[train_size:]
    
    best_aic = np.inf
    best_order = None
    best_model = None
    for p,d,q in itertools.product(range(4), range(2), range(4)):
        try:
            model = ARIMA(train, order=(p,d,q))
            model_fit = model.fit()
            if model_fit.aic < best_aic:
                best_aic = model_fit.aic
                best_order = (p,d,q)
                best_model = model_fit
        except:
            continue
    
    test_forecast = best_model.forecast(steps=len(test))
    rmse = np.sqrt(mean_squared_error(test, test_forecast))
    
    future_dates = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    future_forecast = best_model.forecast(steps=forecast_steps)
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_forecast})
    
    return forecast_df, rmse

# Создание модели LSTM
def train_lstm_model(df, forecast_steps=10, look_back=10, epochs=50, batch_size=16):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').set_index('date')
    series = df['close'].astype(float).interpolate().values.reshape(-1,1)
    
    scaler = MinMaxScaler()
    series_scaled = scaler.fit_transform(series)
    
    train_size = int(len(series_scaled)*0.8)
    train, test = series_scaled[:train_size], series_scaled[train_size:]
    
    def create_dataset(dataset, look_back):
        X, y = [], []
        for i in range(len(dataset)-look_back):
            X.append(dataset[i:i+look_back,0])
            y.append(dataset[i+look_back,0])
        return np.array(X), np.array(y)
    
    X_train, y_train = create_dataset(train, look_back)
    X_test, y_test = create_dataset(np.vstack((train[-look_back:], test)), look_back)
    X_train = X_train.reshape((X_train.shape[0], look_back,1))
    X_test = X_test.reshape((X_test.shape[0], look_back,1))
    
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back,1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled)
    y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
    
    min_len = min(len(y_pred), len(y_test_real))
    y_pred = y_pred[:min_len]
    y_test_real = y_test_real[:min_len]
    rmse = np.sqrt(np.mean((y_test_real - y_pred)**2))
    
    last_seq = series_scaled[-look_back:]
    future_forecast = []
    seq = last_seq.copy()
    for _ in range(forecast_steps):
        x_input = seq.reshape((1, look_back,1))
        yhat = model.predict(x_input)[0,0]
        future_forecast.append(yhat)
        seq = np.append(seq[1:], yhat).reshape(-1,1)
    future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1,1))
    
    future_dates = pd.date_range(start=df.index[-1]+pd.Timedelta(days=1), periods=forecast_steps, freq='B')
    forecast_df = pd.DataFrame({'date': future_dates, 'forecast': future_forecast.flatten()})
    
    return forecast_df, rmse

# Основной Streamlit интерфейс
st.title("Прогноз цены акций NVDA")
st.markdown("""
    <style>
    /* Общий стиль страницы */
    .main {
        background-color: #fafafa;
        color: #333333;
        font-family: "Segoe UI", Roboto, Arial, sans-serif;
    }

    /* Заголовки */
    h1, h2, h3, h4 {
        color: #1f4e79;
        font-weight: 600;
    }

    /* Таблицы Streamlit */
    .stDataFrame {
        border: 1px solid #ccc !important;
        border-radius: 6px;
        overflow: hidden !important;
        margin-top: 0.5rem;
        margin-bottom: 1.5rem;
    }

    /* Центрирование и отступы основного контейнера */
    .block-container {
        padding-top: 2.5rem;  
        padding-bottom: 1rem;
        max-width: 1300px;
        margin: auto;
    }

    /* Разделители */
    hr {
        border: 0;
        border-top: 1px solid #cccccc;
        margin: 2rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Основной контент 
df_nvda = load_data()
st.subheader("Данные NVDA (последние 10 строк)")
st.dataframe(df_nvda.tail(10), use_container_width=False, height=300)

conn = create_db(df_nvda)

if conn:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Объем торгов по годам")
        st.dataframe(query_db(conn, "annual_volume"), use_container_width=False, height=250)

        st.subheader("Средняя цена закрытия по месяцам (2025)")
        st.dataframe(query_db(conn, "monthly_avg_close_2025"), use_container_width=False, height=250)

    with col2:
        st.subheader("Объем торгов по месяцам (2025)")
        st.dataframe(query_db(conn, "monthly_volume_2025"), use_container_width=False, height=250)

        st.subheader("Месячная доходность (2025, %)")
        st.dataframe(query_db(conn, "monthly_return_2025"), use_container_width=False, height=250)

    conn.close()

# Модели ARINA и LSTM
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Настройки прогноза")
col1, col2 = st.columns([1, 3])
with col1:
    forecast_steps = st.selectbox(
        "Горизонт прогноза (дней):",
        [10, 20, 30, 60],
        index=1
    )
with col2:
    st.write("Выберите количество дней, на которые нужно построить прогноз. "
             "Модели будут пересчитаны автоматически.")

with st.spinner("Обучение моделей..."):
    forecast_ar, rmse_ar = train_arima_model(df_nvda, forecast_steps=forecast_steps)
    forecast_lstm, rmse_lstm = train_lstm_model(df_nvda, forecast_steps=forecast_steps)

# метрики 
st.markdown("<hr>", unsafe_allow_html=True)
st.subheader("Результаты моделей")
col1, col2 = st.columns(2)
with col1:
    st.metric(label="RMSE ARIMA", value=f"{rmse_ar:.2f}")
with col2:
    st.metric(label="RMSE LSTM", value=f"{rmse_lstm:.2f}")

st.markdown("<hr>", unsafe_allow_html=True)

# переключатель отображения
mode = st.radio(
    "Выберите тип визуализации:",
    ["ARIMA", "LSTM", "Сравнение моделей"],
    horizontal=True,
)

# Модель ARIMA
if mode == "ARIMA":
    st.subheader("Прогноз ARIMA")
    fig_arima = go.Figure()
    fig_arima.add_trace(go.Scatter(
        x=pd.to_datetime(df_nvda['date']),
        y=df_nvda['close'],
        mode='lines',
        name='Исторические данные',
        line=dict(color="#1f77b4", width=2)
    ))
    fig_arima.add_trace(go.Scatter(
        x=forecast_ar['date'],
        y=forecast_ar['forecast'],
        mode='lines+markers',
        name='ARIMA прогноз',
        line=dict(color="#ff7f0e", dash='dot')
    ))
    fig_arima.update_layout(
        title=f"Прогноз NVDA — модель ARIMA ({forecast_steps} дней вперёд)",
        xaxis_title="Дата",
        yaxis_title="Цена закрытия",
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_arima, use_container_width=True)

# Модель LSTM 
elif mode == "LSTM":
    st.subheader("Прогноз LSTM")
    fig_lstm = go.Figure()
    fig_lstm.add_trace(go.Scatter(
        x=pd.to_datetime(df_nvda['date']),
        y=df_nvda['close'],
        mode='lines',
        name='Исторические данные',
        line=dict(color="#1f77b4", width=2)
    ))
    fig_lstm.add_trace(go.Scatter(
        x=forecast_lstm['date'],
        y=forecast_lstm['forecast'],
        mode='lines+markers',
        name='LSTM прогноз',
        line=dict(color="#2ca02c", dash='dot')
    ))
    fig_lstm.update_layout(
        title=f"Прогноз NVDA — модель LSTM ({forecast_steps} дней вперёд)",
        xaxis_title="Дата",
        yaxis_title="Цена закрытия",
        hovermode="x unified",
        template="plotly_white",
        height=550,
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_lstm, use_container_width=True)

# сравнение моделей ARIMA и LSTM
else:
    st.subheader("Сравнение моделей ARIMA и LSTM")
    fig_compare = go.Figure()
    fig_compare.add_trace(go.Scatter(
        x=pd.to_datetime(df_nvda['date']),
        y=df_nvda['close'],
        mode='lines',
        name='Исторические данные',
        line=dict(color="#1f77b4", width=2)
    ))
    fig_compare.add_trace(go.Scatter(
        x=forecast_ar['date'],
        y=forecast_ar['forecast'],
        mode='lines+markers',
        name='ARIMA прогноз',
        line=dict(color="#ff7f0e", dash='dot')
    ))
    fig_compare.add_trace(go.Scatter(
        x=forecast_lstm['date'],
        y=forecast_lstm['forecast'],
        mode='lines+markers',
        name='LSTM прогноз',
        line=dict(color="#2ca02c", dash='dot')
    ))
    fig_compare.update_layout(
        title=f"Сравнение прогнозов ARIMA и LSTM для NVDA ({forecast_steps} дней вперёд)",
        xaxis_title="Дата",
        yaxis_title="Цена закрытия",
        hovermode="x unified",
        template="plotly_white",
        height=600,
        legend=dict(x=0.02, y=0.98)
    )
    st.plotly_chart(fig_compare, use_container_width=True)
