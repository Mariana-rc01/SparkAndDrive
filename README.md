# NYC Yellow Taxi 2020

## Overview

This project presents an end-to-end Big Data analytics pipeline built on the **NYC Yellow Taxi Trip Records (2020)** dataset published by the NYC Taxi & Limousine Commission (TLC).

Using **Apache Spark** deployed on **Google Cloud Dataproc**, the project explores large-scale urban mobility data through:

* Distributed data processing
* Exploratory data analysis
* Spark SQL analytical querying
* Machine Learning with Spark MLlib
* Runtime and scalability evaluation across multiple Spark configurations
* Interactive dashboard visualization with Streamlit

The study focuses on the impact of the COVID-19 pandemic on taxi mobility patterns throughout 2020, covering pre-pandemic activity, lockdown collapse, and gradual recovery phases.

## Technologies Used

### Big Data & Processing

* Apache Spark 3.3.2
* PySpark
* Spark SQL
* Google Cloud Dataproc

### Machine Learning

* Spark MLlib
* Regression Models:

  * Linear Regression
  * Generalized Linear Regression
  * Decision Tree Regressor
  * Random Forest Regressor
  * Gradient Boosted Trees

### Visualization & Dashboard

* Streamlit
* Plotly
* Pandas

### Dataset Format

* Apache Parquet

## Dataset

Dataset source:

* NYC TLC Yellow Taxi Trip Records (2020)

The dataset contains:

* 24+ million taxi trips
* Pickup and dropoff timestamps
* Passenger count
* Trip distance
* Fare information
* Payment type
* Pickup/dropoff locations

Additional spatial enrichment was performed using the TLC Taxi Zone Lookup table.

## Project Structure

```text
project-root/
│
├── data/
│   ├── 2020/
│   ├── results/
│   │   ├── models/
│   │   └── queries/
│   └── taxi_zones/
│
├── ml_predictions/
│   ├── linear_regression.ipynb
│   ├── glm.ipynb
│   ├── decision_tree.ipynb
│   ├── random_forest.ipynb
│   └── gbt.ipynb
│
├── web-page/
│   └── app.py
│
├── profiling&queries.ipynb
├── results&performance.ipynb
│
├── queries_results.json
├── models_results.json
│
├── report.typ
├── refs.bib
│
└── README.md
```

## Data Processing Pipeline

The project follows a complete Spark-based pipeline:

### 1. Data Ingestion

* Monthly Parquet files are loaded into Spark DataFrames
* Schema normalization is applied

### 2. Data Cleaning

The following issues were handled:

* Missing values
* Invalid timestamps
* Negative distances
* Outliers
* Duplicate records
* Logical inconsistencies

Approximately **95% of the original data** was retained after cleaning.

### 3. Feature Engineering

New features include:

* Pickup hour
* Day of week
* Time of day
* Trip duration
* Average speed
* Cost per mile

### 4. Analytical Queries

Multiple analytical dashboards were developed:

* Billing & tips
* Geographic demand patterns
* Temporal dynamics
* Airport activity
* Operational efficiency

### 5. Machine Learning

Two regression tasks:

* `total_amount`
* `trip_duration_min`

Models were evaluated using:

* RMSE
* MAE
* MSE
* R²

### 6. Scalability Analysis

Different Spark execution scenarios were tested:

* Balanced
* High parallelism
* Memory optimized
* AQE only
* CPU heavy
* Stress test

## Running the Project

## Requirements

Install the required Python packages:

```bash
pip install pyspark pandas plotly streamlit
```

Depending on the notebooks used, additional packages may also be required.

## Running the Notebooks

### Data Profiling & Queries

```bash
jupyter notebook profiling&queries.ipynb
```

### Performance & Scalability Analysis

```bash
jupyter notebook results&performance.ipynb
```

### Machine Learning Models

Each model is implemented in an individual notebook inside:

```text
ml_predictions/
```

Run them independently using Jupyter Notebook.

## Running the Dashboard

The interactive dashboard is located in:

```text
web-page/app.py
```

### Start the Streamlit application

From the `web-page` directory:

```bash
streamlit run app.py
```

## Authors

* Beatriz Iara Nunes Silva
* Leonor Filipe Fragoso e Santos
* Luana Filipa de Matos Lima
* Mariana Rocha Cristino
* Patrícia Crespo da Silva

Faculty of Engineering — University of Porto (FEUP)
