# 🏙️ Socio-Economic Downtrend & Gentrification Risk Predictor

A **web-based dashboard** to analyze real estate trends and predict **socio-economic risks** across Indian states, cities, and neighborhoods. This project leverages historical property data, socio-economic indicators, and **machine learning** to identify areas at risk of gentrification or economic downtrend.

---

## 🔹 Features

- Interactive **dashboard** using Streamlit.
- Explore **real estate price trends** per neighborhood, city, and state.
- Visualize **socio-economic indicators** over time, including:
  - Infrastructure Growth
  - Crime Rate
  - Education Access
  - Healthcare Access
  - Employment Rate
  - Population Growth
- Compare **neighborhood trends** within a city.
- Predict **next year’s price per sqft** using Linear Regression.
- Predict **socio-economic risk** based on user-input indicators using Random Forest classifier.
- Visualize **prediction probabilities** for risk categories.

---

## 📊 Dataset

- The project uses a **real estate and socio-economic dataset**: `indian_realestate_socioeconomic.csv`.
- Key columns:
  - `State`, `City`, `Neighborhood` – Geographic information
  - `Year` – Time period of the record
  - `PricePerSqft` – Property price per square foot
  - `InfraGrowth`, `CrimeRate`, `EducationAccess`, `HealthcareAccess`, `EmploymentRate`, `PopulationGrowth` – Socio-economic indicators
  - `Risk` – Risk category label (Low, Medium, High, etc.)

> Note: Ensure the dataset is in the **same directory** as the Streamlit app.

---
