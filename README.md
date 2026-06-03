# Urban Microclimate Digital Twin for Hyper-Local Heatwave Prediction

## Overview

Urban heatwaves are becoming increasingly severe in densely populated cities. This project presents an AI-powered Urban Microclimate Digital Twin capable of forecasting short-term temperature trends, identifying heat-risk zones, detecting urban heat pockets, and visualizing spatial heat distribution across Delhi.

The system combines temporal forecasting using Random Forest Regression with spatial refinement through Graph Convolutional Networks (GCNs), providing hyper-local climate insights through an interactive Streamlit dashboard.

## Key Features

- Hyper-local temperature forecasting
- Heat Stress Index (HSI) calculation
- Urban Heat Island (UHI) analysis
- Zone-level heat risk classification
- Heat pocket detection using clustering
- Spatial temperature refinement using Graph Neural Networks
- MODIS-inspired Land Surface Temperature visualization
- Interactive Streamlit dashboard
- Automated heatwave alert system

## Dataset

Source: Open-Meteo Archive API

Location: Delhi, India

Time Period:
January 2024 – June 2026

Records:
21,240+ hourly weather observations

Features:
- Temperature
- Relative Humidity
- Wind Speed
- Hour
- Day of Year
- Month

## Model Performance

| Model | MAE (°C) | RMSE (°C) | R² |
|---------|---------|---------|---------|
| RF Temporal | 0.469 | 0.705 | 0.9937 |
| RF Spatial | 0.800 | 1.034 | 0.9811 |
| RF + GCN Hybrid | 0.560 | 0.758 | 0.9898 |

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- PyTorch Geometric
- Streamlit
- Plotly
- Open-Meteo API
- GeoPandas

## Project Structure

- `app.py` → Interactive Streamlit Dashboard
- `train_hybrid.py` → Random Forest + GCN Training Pipeline
- `fetch_historical_data.py` → Open-Meteo Data Collection
- `saved_models/` → Trained ML Models
- `datasets/` → Historical Weather Data
- `screenshots/` → Dashboard Visualizations
- `requirements.txt` → Project Dependencies

## Future Enhancements

- Real-time IoT sensor integration
- Live weather ingestion pipeline
- Satellite imagery integration
- Deep learning forecasting models
- Mobile application deployment
- Smart city heatwave alert network

## Author

Abhinav Gaur

B.Tech Computer Science Engineering (Big Data Analytics)

SRM Institute of Science and Technology

