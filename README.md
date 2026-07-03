# 📊 AI-Powered Retail Analytics Decision Engine

An end-to-end AI-powered retail analytics platform that combines multiple machine learning and time-series forecasting models to predict retail sales, compare forecasting performance, and generate automated business recommendations through an interactive Streamlit dashboard.

The project demonstrates the complete machine learning workflow, including data preprocessing, feature engineering, model training, evaluation, visualization, and business decision support. By integrating multiple forecasting techniques into a unified application, the system enables users to compare model performance and make data-driven retail decisions.

---

# Overview

Accurate sales forecasting is critical for inventory planning, pricing strategies, demand forecasting, and business growth. Traditional forecasting systems often rely on a single predictive model, making them less robust when customer demand patterns change.

This project addresses that challenge by building a multi-model retail forecasting and decision-support system that:

- Forecasts future retail sales using multiple machine learning and time-series models.
- Compares forecasting models using standard evaluation metrics.
- Generates automated business recommendations based on forecasted demand.
- Provides an interactive Streamlit dashboard for visualization and decision-making.
- Exports recommendation reports for further business analysis.

Rather than depending on a single forecasting approach, the system allows users to evaluate multiple models under the same dataset and select the most suitable model based on performance.

---

# Problem Statement

Retail businesses depend on accurate sales forecasts to optimize inventory management, pricing strategies, and promotional planning. Forecasting demand using a single predictive model can produce inconsistent results because different algorithms capture different demand patterns.

This project addresses that challenge by providing a unified forecasting platform that compares multiple machine learning and time-series forecasting models, evaluates their predictive performance, and generates actionable business recommendations through an interactive dashboard.

---

# Design Questions

The system was designed to answer the following questions:

- Which forecasting model provides the best predictive performance for the given retail dataset?
- How do classical machine learning models compare with statistical and deep learning approaches?
- How can forecasting results be translated into actionable business recommendations?
- How can model evaluation be presented in an intuitive dashboard for business users?
- How can forecasting workflows be automated into a single end-to-end application?

--- --- ---

# Key Features

- Multi-model sales forecasting
- Automated model comparison
- Forecast accuracy evaluation (MAE, RMSE, MAPE)
- Interactive Streamlit dashboard
- Business recommendation engine
- Inventory recommendation system
- Sales trend visualization
- Weekly and monthly forecasting
- Revenue estimation
- CSV export of generated recommendations

--- --- ---

# Tech Stack

### Programming Language
- Python

### Machine Learning & Forecasting
- Prophet
- XGBoost
- Random Forest
- TensorFlow / Keras (LSTM)
- Scikit-learn

### Data Processing
- Pandas
- NumPy

### Data Visualization
- Plotly Express

### Web Application
- Streamlit

### Development Environment
- Visual Studio Code
- Git
- GitHub

--- --- ---

# System Architecture

                         Retail Sales Dataset
                                  │
                                  ▼
                    Data Loading & Preprocessing
                                  │
                                  ▼
                     Category-wise Data Aggregation
                                  │
                                  ▼
                        Feature Engineering
                                  │
                                  ▼
                     Forecasting Model Selection
            ┌────────────┬────────────┬────────────┬────────────┐
            ▼            ▼            ▼            ▼
        Prophet      XGBoost    Random Forest      LSTM
            └────────────┴────────────┴────────────┴────────────┘
                                  │
                                  ▼
                        Forecast Generation
                                  │
                                  ▼
                    Model Performance Evaluation
                       (MAE • RMSE • MAPE)
                                  │
                                  ▼
                      Best Model Identification
                                  │
                                  ▼
               Business Recommendation Generation
                                  │
                                  ▼
                  Interactive Streamlit Dashboard
                                  │
                                  ▼
                  CSV Export & Summary Download

--- --- ---

# Machine Learning Models

The platform implements four forecasting approaches to enable performance comparison across different machine learning paradigms.

| Model | Purpose |
|-------|---------|
| **Prophet** | Time-series forecasting that captures trends and seasonality in retail sales data. |
| **XGBoost** | Gradient boosting algorithm capable of modelling complex nonlinear relationships. |
| **Random Forest** | Ensemble learning algorithm that combines multiple decision trees for robust forecasting. |
| **LSTM** | Deep learning model designed to learn sequential and temporal patterns from historical sales data. |

Each forecasting model is evaluated independently, allowing users to compare performance before selecting the most suitable model for the given dataset.

--- --- ---

# Model Evaluation

Forecast accuracy is evaluated using three widely adopted regression metrics:

- **MAE (Mean Absolute Error)** – Measures the average prediction error.
- **RMSE (Root Mean Squared Error)** – Penalizes larger prediction errors and is used to identify the best-performing model.
- **MAPE (Mean Absolute Percentage Error)** – Measures prediction error as a percentage of the actual values.

The application compares all supported forecasting models and highlights the model with the lowest RMSE.

--- --- ---

# Dashboard Features

The Streamlit application provides an interactive interface for exploring retail sales data and forecasting future demand.

Current features include:

- Retail sales dataset upload
- Data exploration dashboard
- Sales trend visualization
- Product category selection
- Multi-model forecasting
- Forecast period selection
- Automated model comparison
- Forecast accuracy metrics
- Revenue estimation
- Inventory recommendations
- Pricing strategy recommendations
- Promotion strategy recommendations
- Weekly sales forecasts
- Monthly sales forecasts
- Cumulative sales visualization
- Strategic business guidance
- Recommendation summary generation
- CSV export of recommendations
- Downloadable recommendation summary

--- --- ---

# Repository Structure

```text
AI-Powered-Retail-Analytics-Decision-Engine/
│
├── app.py                  # Streamlit application
├── model_utils.py         # ML models and forecasting logic
├── requirements.txt       # Dependencies
├── data/                  # Dataset storage
├── exports/               # Generated predictions & outputs
├── README.md              # Project documentation
```

---

# Installation

Clone the repository:


```bash
git clone https://github.com/your-username/AI-Powered-Retail-Analytics-Decision-Engine.git
cd AI-Powered-Retail-Analytics-Decision-Engine
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

# Dataset

The application expects a retail sales dataset containing historical transaction records.

Required columns include:

- Date
- Product Category
- Quantity
- Price per Unit
- Total Amount

After the dataset is uploaded through the Streamlit interface, the application automatically performs preprocessing and category-wise aggregation before training the selected forecasting model.

--- --- ---

# Outputs

The application generates the following outputs:

- Sales forecasts generated using Prophet, XGBoost, Random Forest, and LSTM models.
- Model evaluation metrics including MAE, RMSE, and MAPE.
- Comparative performance reports across all forecasting models.
- Forecast revenue estimates based on user-defined pricing.
- Inventory, pricing, and promotion recommendations.
- Weekly, monthly, and cumulative sales forecast visualizations.
- Interactive business analytics dashboard.
- Downloadable recommendation summaries.
- Exportable CSV reports containing strategic business recommendations.

Generated reports are stored under:


```text
exports/
```
--- --- ---

# Reproducibility

This project supports partial reproducibility:

Level 1 — Analysis

Run evaluation and comparison on existing outputs.

Level 2 — Prediction

Regenerate forecasts using trained models.

Level 3 — Full Pipeline

Retrain models and regenerate all outputs (requires compute resources).

--- --- ---

# Future Improvements

- Confidence intervals for forecast uncertainty
- Automated hyperparameter tuning
- Feature importance visualization for all models
- Real-time retail data integration
- Cloud deployment
- Docker containerization
- REST API for forecast generation
- CI/CD pipeline for automated deployment
- MLOps-based model monitoring
- LLM-powered business insight generation

--- --- ---

# Contact

If you have questions, suggestions, or feedback about this project, feel free to open an issue in this repository.
