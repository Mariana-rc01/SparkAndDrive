import streamlit as st
import pandas as pd
import plotly.express as px
import json

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(page_title="NYC Taxi — Spark Performance", layout="wide")

# Updated CSS with hover effects for metric boxes
st.markdown("""
    <style>
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
    }
    .metric-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.1);
        border-color: #AEC6CF;
    }
    .metric-label {
        font-size: 11px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 5px;
    }
    .metric-value {
        font-size: 22px; font-weight: 600; color: #111;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    try:
        with open('data/results/queries/queries_results.json', 'r', encoding='utf-8') as f:
            queries_data = json.load(f)

        with open('data/results/models/models_results.json', 'r', encoding='utf-8') as f:
            models_data = json.load(f)

        return queries_data, models_data
    except FileNotFoundError as e:
        st.error(f"Error: File not found. Details: {e}")
        st.stop()
    except json.JSONDecodeError:
        st.error("Error: Invalid JSON file.")
        st.stop()

QUERIES_DATA, MODELS_DATA = load_data()

# Updated Color Palettes with Pastel Tones
SCENARIO_COLORS = {
  'baseline':'#AEC6CF',           # Pastel Blue
  'balanced':'#B3E2CD',           # Pastel Green
  'high_parallelism':'#FDE0A3',   # Pastel Yellow
  'memory_optimized':'#F4B183',   # Pastel Orange
  'stress_test':'#FFB3BA',        # Pastel Red
  'low_overhead':'#CBAACB',       # Pastel Purple
  'aqe_only':'#C8E6C9',           # Pastel Mint
  'cpu_heavy':'#FFDFD3'           # Pastel Peach
}

MODEL_COLORS = {
  'Linear Regression':'#AEC6CF',  # Pastel Blue
  'Decision Tree':'#B3E2CD',      # Pastel Green
  'GLM':'#FDE0A3',                # Pastel Yellow
  'Random Forest':'#CBAACB',      # Pastel Purple
  'GBT':'#FFB3BA'                 # Pastel Red
}

MODEL_NAME_MAP = {
    "LinearRegression_Ridge_Std": "Linear Regression",
    "DecisionTreeRegressor": "Decision Tree",
    "GeneralizedLinearModel": "GLM",
    "RandomForestRegressor": "Random Forest",
    "GBTRegressor": "GBT"
}

# ==========================================
# 3. HEADER & SCENARIOS TABLE
# ==========================================
st.title("NYC Taxi 2020 — Spark Performance Dashboard")
st.markdown("<span style='color:#888;'>Analysis of query execution times and Machine Learning results across cluster configurations.</span>", unsafe_allow_html=True)

scenarios_info = {}
for run in QUERIES_DATA.get("runs", []):
    mode = run["spark_mode"]
    if mode not in scenarios_info and "spark_config" in run:
        scenarios_info[mode] = run["spark_config"]

if scenarios_info:
    df_config = pd.DataFrame.from_dict(scenarios_info, orient='index').reset_index()
    df_config = df_config.rename(columns={
        "index": "Cluster Scenario",
        "executor_instances": "Executors",
        "executor_cores": "Cores per Executor",
        "executor_memory": "RAM per Executor",
        "shuffle_partitions": "Shuffle Partitions",
        "adaptive": "AQE Enabled"
    })

    with st.expander("View technical configurations for each Cluster Scenario"):
        st.markdown("Below are the exact resources allocated to Apache Spark in each scenario:")
        st.dataframe(df_config, use_container_width=True, hide_index=True)

st.divider()

# ==========================================
# 4. TABS
# ==========================================
tab_queries, tab_models = st.tabs(["Query Times", "ML Models"])

# --- TAB 1: QUERIES ---
with tab_queries:
    st.subheader("SQL Queries Analysis")

    col1, col2, col3 = st.columns(3)

    section_options = {"Q1": "Section 1 — Billing & Tips", "Q2": "Section 2 — Geographic", "Q3": "Section 3 — Temporal", "Q4": "Section 4 — Airports", "Q5": "Section 5 — Efficiency"}
    q_sections = col1.multiselect("Section(s)", options=list(section_options.keys()), format_func=lambda x: section_options[x], default=["Q1"])

    all_q_scenarios = list(dict.fromkeys(r["spark_mode"] for r in QUERIES_DATA.get("runs", [])))
    q_scenarios = col2.multiselect("Cluster Scenario(s)", options=all_q_scenarios, default=all_q_scenarios[:2] if len(all_q_scenarios) > 1 else all_q_scenarios)

    metric_options = {"API Time (s)": "API Time (s)", "SQL Time (s)": "SQL Time (s)", "Time Delta (SQL - API)": "Time Delta (SQL − API) (s)"}
    q_metric = col3.selectbox("Metric", options=list(metric_options.keys()), format_func=lambda x: metric_options[x])

    if not q_scenarios or not q_sections:
        st.info("Please select at least one Section and one Scenario to view data.")
    else:
        rows = []
        for run in QUERIES_DATA.get("runs", []):
            scen = run["spark_mode"]
            if scen in q_scenarios:
                for m in run.get("metrics", []):
                    if any(m["Query Description"].startswith(s) for s in q_sections):
                        short_desc = m["Query Description"].split(": ")[1] if ": " in m["Query Description"] else m["Query Description"]
                        rows.append({
                            "Scenario": scen,
                            "Query": short_desc,
                            "Value": m[q_metric],
                            "API Time": m["API Time (s)"],
                            "SQL Time": m["SQL Time (s)"],
                            "Delta": m["Time Delta (SQL - API)"]
                        })

        df_queries = pd.DataFrame(rows)

        if not df_queries.empty:
            avg_api = df_queries["API Time"].mean()
            avg_sql = df_queries["SQL Time"].mean()
            max_time = df_queries[["API Time", "SQL Time"]].max().max()
            faster_api = (df_queries["Delta"] < 0).sum()

            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            kpi1.markdown(f"<div class='metric-box'><div class='metric-label'>Avg API Time</div><div class='metric-value'>{avg_api:.2f}s</div></div>", unsafe_allow_html=True)
            kpi2.markdown(f"<div class='metric-box'><div class='metric-label'>Avg SQL Time</div><div class='metric-value'>{avg_sql:.2f}s</div></div>", unsafe_allow_html=True)
            kpi3.markdown(f"<div class='metric-box'><div class='metric-label'>Slowest Query</div><div class='metric-value'>{max_time:.2f}s</div></div>", unsafe_allow_html=True)
            kpi4.markdown(f"<div class='metric-box'><div class='metric-label'>API Faster than SQL</div><div class='metric-value'>{faster_api} / {len(df_queries)}</div></div>", unsafe_allow_html=True)

            st.write("")

            df_queries = df_queries.sort_values(by=["Query", "Scenario"], ascending=[False, True])

            unique_queries = len(df_queries["Query"].unique())
            chart_height = max(400, unique_queries * 35 * len(q_scenarios))

            fig_q = px.bar(
                df_queries,
                x="Value",
                y="Query",
                color="Scenario",
                barmode="group",
                orientation='h',
                color_discrete_map=SCENARIO_COLORS,
                height=chart_height
            )

            fig_q.update_layout(
                xaxis_title=metric_options[q_metric],
                yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_q.update_xaxes(gridcolor='rgba(0,0,0,0.05)')

            st.plotly_chart(fig_q, use_container_width=True)


# --- TAB 2: ML MODELS ---
with tab_models:
    st.subheader("ML Models Performance")

    col1, col2, col3, col4 = st.columns(4)

    target_options = {"total_amount": "Total Amount", "trip_duration_min": "Trip Duration"}
    m_targets = col1.multiselect("Target Variable", options=list(target_options.keys()), format_func=lambda x: target_options[x], default=list(target_options.keys()))

    m_models = col2.multiselect("Model(s)", options=list(MODEL_NAME_MAP.keys()), format_func=lambda x: MODEL_NAME_MAP[x], default=list(MODEL_NAME_MAP.keys())[:2])

    all_m_scenarios = list(dict.fromkeys(r["spark_mode"] for r in MODELS_DATA.get("runs", [])))
    m_scenarios = col3.multiselect("Cluster Scenario(s)", options=all_m_scenarios, default=all_m_scenarios[:3] if len(all_m_scenarios) > 2 else all_m_scenarios)

    m_metric_options = {"rmse": "RMSE", "r2": "R²", "mae": "MAE", "execution_time_seconds": "Execution Time (s)"}
    m_metric = col4.selectbox("ML Metric", options=list(m_metric_options.keys()), format_func=lambda x: m_metric_options[x])

    if not m_targets or not m_models or not m_scenarios:
        st.info("Please select at least one Target Variable, one Model, and one Scenario.")
    else:
        chart_data = []
        for run in MODELS_DATA.get("runs", []):
            scenario = run.get("spark_mode")
            if scenario in m_scenarios:
                for result in run.get("model_results", []):
                    mod_id = result.get("model_name")
                    tgt_id = result.get("target")

                    if mod_id in m_models and tgt_id in m_targets:
                        val = result.get("execution_time_seconds") if m_metric == "execution_time_seconds" else result.get("metrics", {}).get(m_metric)
                        if val is not None:
                            label_name = MODEL_NAME_MAP[mod_id]
                            if len(m_targets) > 1:
                                label_name = f"{label_name} ({target_options[tgt_id]})"

                            chart_data.append({
                                "Scenario": scenario,
                                "Model/Target": label_name,
                                "Value": round(val, 4),
                                "BaseModel": MODEL_NAME_MAP[mod_id]
                            })

        if chart_data:
            df_models = pd.DataFrame(chart_data)

            best_val = df_models["Value"].min()
            worst_val = df_models["Value"].max()
            avg_val = df_models["Value"].mean()

            k1, k2, k3, k4 = st.columns(4)
            k1.markdown(f"<div class='metric-box'><div class='metric-label'>Best {m_metric_options[m_metric]}</div><div class='metric-value'>{best_val:.3f}</div></div>", unsafe_allow_html=True)
            k2.markdown(f"<div class='metric-box'><div class='metric-label'>Worst {m_metric_options[m_metric]}</div><div class='metric-value'>{worst_val:.3f}</div></div>", unsafe_allow_html=True)
            k3.markdown(f"<div class='metric-box'><div class='metric-label'>Average</div><div class='metric-value'>{avg_val:.3f}</div></div>", unsafe_allow_html=True)
            k4.markdown(f"<div class='metric-box'><div class='metric-label'>Data Points</div><div class='metric-value'>{len(df_models)}</div></div>", unsafe_allow_html=True)

            st.write("")

            color_mapping = {row["Model/Target"]: MODEL_COLORS.get(row["BaseModel"], '#333') for _, row in df_models.iterrows()}

            fig_m = px.bar(
                df_models,
                x="Value",
                y="Scenario",
                color="Model/Target",
                orientation='h',
                barmode='group',
                color_discrete_map=color_mapping,
                height=max(400, len(m_scenarios) * max(1, len(df_models["Model/Target"].unique())) * 30)
            )

            fig_m.update_layout(
                xaxis_title=m_metric_options[m_metric],
                yaxis_title="",
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            fig_m.update_xaxes(gridcolor='rgba(0,0,0,0.05)')

            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.warning("No model data found for the selected filters.")
