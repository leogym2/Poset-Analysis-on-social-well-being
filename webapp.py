import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =====================
# --- Load Data ---
# =====================
data_europe = pd.read_csv("data_europe.csv")
w_1s = pd.read_csv("ls_weights.csv", index_col=0)
w_1s = w_1s.iloc[:, 0]  # convert to Series
w_1s.index.name = None

comparison_full = pd.read_csv("comparison_full.csv")
comparison_full = comparison_full.merge(
    data_europe[["Region", "Country"]],
    left_on="region", right_on="Region", how="left"
)
comparison_full.drop(columns=["Region"], inplace=True)

features = w_1s.index.tolist()
ls_col = "Life satisfaction"
all_countries = sorted(comparison_full["Country"].dropna().unique())

# =====================
# --- Streamlit Setup ---
# =====================
st.set_page_config(page_title="Poset Analysis on Social Well Being", layout="wide")

# =====================
# --- PROJECT TITLE ---
# =====================
st.markdown("# Poset Analysis on Social Well Being")
st.markdown("""
This interactive project analyses **social well-being** across European regions.  
The project is divided into two main parts:

1. **Objective vs Subjective Rankings:** Compare Average Rank (AR) with Life Satisfaction (LS).  
2. **LS-Calibrated Composite Scores:** Adjust indicator weights interactively and explore top/flop regions.
""")
st.markdown("---")

# =====================
# --- PART 1 ---
# =====================
st.markdown("## Part 1: Objective vs Subjective Well-Being")
st.markdown("""
**Scatter plot description:**  
This chart compares each region's **Average Rank (AR)**, derived from objective social well-being indicators,  
with **Life Satisfaction (LS)**, a subjective measure reported by residents.

- Points **above the diagonal**: LS higher than expected from AR  
- Points **below the diagonal**: LS lower than expected from AR  
- Points **on the diagonal**: perfect alignment between objective and subjective measures
""")

# Country selection with defaults Italy and Germany
selected_countries = st.multiselect(
    "Select countries to display:",
    options=all_countries,
    default=["Italy", "Germany"],
    key="country_selector_part1"
)

plot_df_visible = comparison_full[comparison_full["Country"].isin(selected_countries)]
top3_ls = plot_df_visible.nsmallest(3, "LS_rank")
bottom3_ls = plot_df_visible.nlargest(3, "LS_rank")

# Scatter plot AR vs LS
fig_ar_ls = px.scatter(
    plot_df_visible,
    x="AR_rank",
    y="LS_rank",
    hover_data={"region": True, "AR_rank": True, "LS_rank": True},
    color_discrete_sequence=["white"],
    labels={"AR_rank": "Average Rank (AR)", "LS_rank": "Life Satisfaction Rank (LS)"},
    title="Comparison of AR vs Life Satisfaction Ranks"
)
max_rank = max(plot_df_visible["AR_rank"].max(), plot_df_visible["LS_rank"].max())
fig_ar_ls.add_shape(
    type="line",
    x0=1, y0=1, x1=max_rank, y1=max_rank,
    line=dict(color="blue", dash="dash")
)
fig_ar_ls.update_yaxes(autorange="reversed")
fig_ar_ls.update_layout(legend_title_text="", title_font=dict(size=22))

st.markdown("### Scatter Plot: AR vs LS")
st.markdown("Each point represents a region. Hover to see AR and LS ranks.")
st.plotly_chart(fig_ar_ls, use_container_width=True)

# Top/Bottom 3 LS tables
st.markdown("### Top 3 regions by Life Satisfaction")
st.markdown("Regions with the highest subjective well-being among the selected countries. Compare their AR ranks.")
st.dataframe(top3_ls[["region","Country","LS_rank","AR_rank"]], use_container_width=True)

st.markdown("### Bottom 3 regions by Life Satisfaction")
st.markdown("Regions with the lowest Life Satisfaction among selected countries. Observe differences from AR ranks.")
st.dataframe(bottom3_ls[["region","Country","LS_rank","AR_rank"]], use_container_width=True)

# =====================
# --- PART 2 ---
# =====================
st.markdown("---")
st.markdown("## Part 2: LS-Calibrated Composite Scores")
st.markdown("""
This part allows you to **adjust the contribution of each social well-being indicator** to a composite score  
that is aligned with Life Satisfaction (LS).  

- Higher weights indicate stronger influence in approximating LS.  
- You can explore how the composite ranking of regions changes as you modify the weights.
""")

# --- Histogram LS-calibrated weights ---
st.markdown("### Original LS-calibrated Indicator Weights")
st.markdown("Histogram showing pre-calibrated influence of each indicator on the LS-aligned composite score.")

fig_weights = px.bar(
    w_1s.sort_values(ascending=False),
    x=w_1s.sort_values(ascending=False).index,
    y=w_1s.sort_values(ascending=False).values,
    labels={"x": "Indicator", "y": "Weight"},
    text=w_1s.sort_values(ascending=False).round(3).values,
    title="LS-calibrated Indicator Weights"
)
fig_weights.update_traces(textposition='outside', marker_color='steelblue')
fig_weights.update_layout(yaxis_tickformat=".0%", xaxis_tickangle=-45)
st.plotly_chart(fig_weights, use_container_width=True)

# --- Helper: redistribute weights ---
def adjust_weights_all(slider_idx, new_value, weights):
    weights = weights.copy()
    delta = new_value - weights[slider_idx]
    others_idx = [i for i in range(len(weights)) if i != slider_idx]
    sum_others = weights[others_idx].sum()
    if sum_others == 0:
        for i in others_idx:
            weights[i] = (1 - new_value) / len(others_idx)
    else:
        for i in others_idx:
            weights[i] -= delta * (weights[i] / sum_others)
    weights = np.clip(weights, 0, 1)
    weights = weights / weights.sum()
    return weights

# --- Sliders for weights ---
st.markdown("### Adjust indicator weights interactively")
st.markdown("Move the sliders to change each indicator's contribution. The sum of weights always remains 1.")
weights_input = w_1s.values.copy()
cols = st.columns(3)
for i, feature in enumerate(features):
    col = cols[i % 3]
    val = col.slider(
        feature,
        min_value=0.0,
        max_value=1.0,
        value=float(weights_input[i]),
        step=0.01,
        key=f"slider_{i}"
    )
    weights_input = adjust_weights_all(i, val, weights_input)

weights_series = pd.Series(weights_input, index=features)

# --- Compute composite scores and ranks ---
X = data_europe[features].apply(pd.to_numeric, errors="coerce")
composite_scores = (X * weights_series).sum(axis=1)
composite_ranks = composite_scores.rank(method="average", ascending=False)

# --- Top 10 / Flop 10 ---
st.markdown("### Top 10 regions (weighted composite)")
top10 = pd.DataFrame({
    "Region": data_europe["Region"],
    "Country": data_europe["Country"],
    "Composite Score": composite_scores,
    "Rank": composite_ranks
}).sort_values("Rank").head(10)
st.markdown("Regions with the highest LS-calibrated composite scores.")
st.dataframe(top10, use_container_width=True)

st.markdown("### Flop 10 regions (weighted composite)")
flop10 = pd.DataFrame({
    "Region": data_europe["Region"],
    "Country": data_europe["Country"],
    "Composite Score": composite_scores,
    "Rank": composite_ranks
}).sort_values("Rank", ascending=False).head(10)
st.markdown("Regions with the lowest LS-calibrated composite scores.")
st.dataframe(flop10, use_container_width=True)


# --- Normalized weights after adjustment ---
st.markdown("### Normalized indicator weights after adjustment")
st.markdown("Final normalized weights. The sum of weights is always 1.")
st.bar_chart(weights_series)
