import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import numpy as np

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("indian_realestate_socioeconomic.csv")
    return df

df = load_data()

st.set_page_config(page_title="Socioeconomic Downtrend Predictor",
                   layout="wide",
                   page_icon="🏙️")

st.title("🏙️ Socioeconomic Downtrend / Gentrification Risk Predictor")
st.markdown("Analyze real estate trends and predict socioeconomic risks across Indian states, cities, and neighborhoods.")

# Sidebar filters
st.sidebar.header("🔎 Filter Options")

states = df["State"].unique()
selected_state = st.sidebar.selectbox("Select State", states)

cities = df[df["State"] == selected_state]["City"].unique()
selected_city = st.sidebar.selectbox("Select City", cities)

neighborhoods = df[(df["State"] == selected_state) & (df["City"] == selected_city)]["Neighborhood"].unique()
selected_neighborhood = st.sidebar.selectbox("Select Neighborhood", neighborhoods)

filtered_df = df[(df["State"] == selected_state) &
                 (df["City"] == selected_city) &
                 (df["Neighborhood"] == selected_neighborhood)]

# --- Price Trend Line Chart ---
st.subheader(f"📈 Price Trend in {selected_neighborhood}, {selected_city} ({selected_state})")
fig_price = px.line(filtered_df,
                    x="Year",
                    y="PricePerSqft",
                    markers=True,
                    title="Price Per Sqft Trend",
                    labels={"PricePerSqft": "Price (₹/sqft)", "Year": "Year"})
st.plotly_chart(fig_price, use_container_width=True)

# --- Risk Display ---
latest_year = filtered_df["Year"].max()
latest_data = filtered_df[filtered_df["Year"] == latest_year]

risk = latest_data["Risk"].values[0]
price = latest_data["PricePerSqft"].values[0]

st.metric(label="📊 Current Risk Category", value=risk)
st.metric(label=f"💰 Price per Sqft in {latest_year}", value=f"₹{price:,.0f}")

# --- Socioeconomic Factors ---
st.subheader("📉 Socio-Economic Indicators Impact")
socio_cols = ["InfraGrowth", "CrimeRate", "EducationAccess", "HealthcareAccess", "EmploymentRate", "PopulationGrowth"]

latest_socio = latest_data[socio_cols].T.reset_index()
latest_socio.columns = ["Factor", "Value"]

fig_socio = px.bar(latest_socio,
                   x="Factor",
                   y="Value",
                   title="Socio-Economic Indicators",
                   text_auto=True,
                   color="Value")
st.plotly_chart(fig_socio, use_container_width=True)

# --- Citywide Trends ---
st.subheader(f"🏘️ Comparison Across Neighborhoods in {selected_city}")

city_df = df[(df["State"] == selected_state) & (df["City"] == selected_city)]
fig_city = px.line(city_df,
                   x="Year",
                   y="PricePerSqft",
                   color="Neighborhood",
                   title=f"Neighborhood Price Comparison - {selected_city}")
st.plotly_chart(fig_city, use_container_width=True)

# --- Socio-Economic Factors over Time ---
st.subheader("📉 Socio-Economic Indicators Over Time")
fig_factors = px.bar(
    filtered_df,
    x="Year",
    y=["InfraGrowth", "CrimeRate", "EducationAccess", "HealthcareAccess", "EmploymentRate", "PopulationGrowth"],
    barmode="group",
    title="Socio-Economic Factors Over Time"
)
st.plotly_chart(fig_factors, use_container_width=True)

# ================================
# 🔮 Machine Learning Prediction
# ================================

# Simple Price Prediction
st.subheader("🤖 Predict Next Year’s Price")
X_lr = filtered_df[["Year"]]
y_lr = filtered_df["PricePerSqft"]
model_lr = LinearRegression().fit(X_lr, y_lr)
pred_next = model_lr.predict([[filtered_df["Year"].max() + 1]])[0]
st.success(f"Predicted Price per sqft in {selected_neighborhood}, {selected_city}, {selected_state} for {filtered_df['Year'].max()+1}: **₹{pred_next:.2f}**")

st.header("🔮 ML-Based Risk Prediction")

# Prepare dataset
X = df[socio_cols]
y = df["Risk"]

# Encode target
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# --- Prediction form ---
st.subheader("📝 Predict Risk for Custom Scenario")

with st.form("prediction_form"):
    infra = st.slider("Infrastructure Growth (0-100)", 0, 100, int(latest_data["InfraGrowth"].values[0]))
    crime = st.slider("Crime Rate (0-100)", 0, 100, int(latest_data["CrimeRate"].values[0]))
    education = st.slider("Education Access (0-100)", 0, 100, int(latest_data["EducationAccess"].values[0]))
    healthcare = st.slider("Healthcare Access (0-100)", 0, 100, int(latest_data["HealthcareAccess"].values[0]))
    employment = st.slider("Employment Rate (0-100)", 0, 100, int(latest_data["EmploymentRate"].values[0]))
    population = st.slider("Population Growth (0-100)", 0, 100, int(latest_data["PopulationGrowth"].values[0]))
    submit = st.form_submit_button("Predict Risk")

if submit:
    input_data = np.array([[infra, crime, education, healthcare, employment, population]])
    prediction = model.predict(input_data)
    prediction_label = le.inverse_transform(prediction)[0]

    probabilities = model.predict_proba(input_data)[0]
    prob_df = pd.DataFrame({"Risk": le.classes_, "Probability": probabilities})

    st.success(f"Predicted Risk: **{prediction_label}**")

    fig_prob = px.bar(prob_df, x="Risk", y="Probability", title="Prediction Probabilities", text_auto=True)
    st.plotly_chart(fig_prob, use_container_width=True)

st.success("✅ Dashboard with ML prediction ready!")
