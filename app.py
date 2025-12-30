import streamlit as st
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from src.visualization import TrafficVisualizer

# Page Config
st.set_page_config(page_title="Smart City Traffic Analysis", page_icon="🏙️", layout="wide")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "traffic_enhanced.csv")
MODELS_DIR = os.path.join(BASE_DIR, "data", "models")
LABEL_ENCODER_PATH = os.path.join(BASE_DIR, "data", "processed", "weather_encoder.pkl")

@st.cache_data
def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['date_time'] = pd.to_datetime(df['date_time'])
        return df
    return None

def load_models():
    try:
        kmeans = joblib.load(os.path.join(MODELS_DIR, "kmeans_traffic.pkl"))
        rf = joblib.load(os.path.join(MODELS_DIR, "rf_traffic_predictor.pkl"))
        encoders = joblib.load(LABEL_ENCODER_PATH)
        return kmeans, rf, encoders
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None, None

def main():
    st.title("🏙️ Smart City Big Data Analytics")
    st.markdown("### Urban Traffic, Population & Environmental Insights")
    
    df = load_data()
    if df is None:
        st.error("Data not found. Please run the data pipeline first.")
        return

    viz = TrafficVisualizer(df)
    kmeans_model, rf_model, encoders = load_models()

    # Sidebar
    st.sidebar.header("Project Info")
    st.sidebar.info("This dashboard visualizes traffic patterns, detects anomalies, and predicts congestion using Big Data techniques.")
    
    st.sidebar.subheader("Filter Data")
    year_filter = st.sidebar.multiselect("Select Year", options=sorted(df['year'].unique()), default=sorted(df['year'].unique())[-1:])
    
    if year_filter:
        df_filtered = df[df['year'].isin(year_filter)]
    else:
        df_filtered = df

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🚦 Traffic Patterns", "🌥️ Weather Impact", "🔮 Prediction"])

    with tab1:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{len(df_filtered):,}")
        col2.metric("Avg Traffic Volume", f"{int(df_filtered['traffic_volume'].mean()):,}")
        col3.metric("Anomalies Detected", f"{df_filtered['is_anomaly'].sum():,}")
        
        st.markdown("#### Hourly Traffic Trend")
        st.pyplot(viz.plot_hourly_traffic())

    with tab2:
        st.subheader("Traffic Clusters & Anomalies")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Traffic Clusters (K-Means)**")
            st.pyplot(viz.plot_cluster_segments())
        with col2:
            st.markdown("**Detected Anomalies (Isolation Forest)**")
            st.pyplot(viz.plot_anomalies())
            
    with tab3:
        st.subheader("Environmental Impact on Traffic")
        st.pyplot(viz.plot_weather_impact())
        
        st.markdown("#### Correlation Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        corr = df_filtered[['traffic_volume', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

    with tab4:
        st.subheader("Traffic Congestion Prediction")
        st.markdown("Enter details to predict traffic volume.")
        
        col1, col2 = st.columns(2)
        with col1:
            p_hour = st.slider("Hour of Day", 0, 23, 12)
            p_day = st.selectbox("Day of Week", range(7), format_func=lambda x: ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][x])
            p_month = st.slider("Month", 1, 12, 6)
        
        with col2:
            p_temp = st.number_input("Temperature (Kelvin)", 200.0, 320.0, 280.0)
            p_rain = st.number_input("Rain (1h mm)", 0.0, 100.0, 0.0)
            p_clouds = st.slider("Cloud Coverage (%)", 0, 100, 20)
            p_weather = st.selectbox("Weather Main", encoders['main'].classes_)
            p_desc = st.selectbox("Weather Description", encoders['desc'].classes_)

        if st.button("Predict Traffic Volume"):
            try:
                # Prepare input
                is_weekend = 1 if p_day >= 5 else 0
                w_main_code = encoders['main'].transform([p_weather])[0]
                w_desc_code = encoders['desc'].transform([p_desc])[0]
                
                features = pd.DataFrame([{
                    'hour': p_hour, 'day_of_week': p_day, 'month': p_month, 'is_weekend': is_weekend,
                    'weather_main_code': w_main_code, 'weather_description_code': w_desc_code,
                    'temp': p_temp, 'rain_1h': p_rain, 'snow_1h': 0, 'clouds_all': p_clouds
                }])
                
                prediction = rf_model.predict(features)[0]
                st.success(f"Predicted Traffic Volume: **{int(prediction)}** cars/hour")
                
                # Context
                if prediction > 5000:
                    st.error("High Congestion Expected! 🚗🟥")
                elif prediction > 3000:
                    st.warning("Moderate Traffic. 🚗🟨")
                else:
                    st.success("Free Flowing Traffic. 🚗🟩")
            except Exception as e:
                st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()
