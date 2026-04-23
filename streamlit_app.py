from pathlib import Path
import os

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib-cache"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


FEATURES = [
    "total_points_diff",
    "game_points_diff",
    "consec_points_diff",
    "g1_score_diff",
    "consec_g1_diff",
    "game_pts_g1_diff",
    "team_one_total_points",
    "team_two_total_points",
    "team_one_game_points",
    "team_two_game_points",
    "team_one_most_consecutive_points",
    "team_two_most_consecutive_points",
    "g1_total",
    "t1_win_pct_g1",
    "round_num",
    "tournament_tier",
    "nb_sets",
]
TARGET = "team_one_wins"

ROUND_ORDER = {
    "Qualification round of 32": 0,
    "Qualification round of 16": 1,
    "Qualification quarter final": 2,
    "Round of 64": 3,
    "Round of 32": 4,
    "Round 1": 4,
    "Round 2": 5,
    "Round 3": 6,
    "Round of 16": 5,
    "Quarter final": 6,
    "Semi final": 7,
    "Final": 8,
}

TIER_ORDER = {
    "BWF Tour Super 100": 1,
    "HSBC BWF World Tour Super 300": 2,
    "HSBC BWF World Tour Super 500": 3,
    "HSBC BWF World Tour Super 750": 4,
    "HSBC BWF World Tour Super 1000": 5,
    "HSBC BWF World Tour Finals": 6,
}

ROUND_LABELS = {value: key for key, value in ROUND_ORDER.items()}
TIER_LABELS = {value: key for key, value in TIER_ORDER.items()}


st.set_page_config(
    page_title="Badminton XAI Match Prediction",
    layout="wide",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1280px;}
    .hero {
        background: linear-gradient(135deg, #083344 0%, #0f766e 52%, #f59e0b 100%);
        color: white;
        padding: 2.2rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .hero h1 {font-size: 2.45rem; margin-bottom: .4rem;}
    .hero p {font-size: 1.05rem; opacity: .94; max-width: 780px;}
    .info-band {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }
    .result-win {
        background: #ecfdf5;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 1rem;
    }
    .result-loss {
        background: #fff7ed;
        border: 1px solid #fdba74;
        border-radius: 8px;
        padding: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def parse_score(value):
    try:
        left, right = str(value).split("-")
        return int(left), int(right)
    except Exception:
        return np.nan, np.nan


@st.cache_data(show_spinner=False)
def load_local_data():
    data_path = Path("ms.csv")
    if data_path.exists():
        return pd.read_csv(data_path)
    return None


@st.cache_data(show_spinner=False)
def prepare_data(raw_df):
    df = raw_df.copy()
    df = df[df["winner"].isin([1, 2])].drop_duplicates().copy()
    df[TARGET] = (df["winner"] == 1).astype(int)

    g1_scores = df["game_1_score"].apply(lambda value: pd.Series(parse_score(value)))
    df[["g1_t1", "g1_t2"]] = g1_scores

    df["total_points_diff"] = df["team_one_total_points"] - df["team_two_total_points"]
    df["game_points_diff"] = df["team_one_game_points"] - df["team_two_game_points"]
    df["consec_points_diff"] = (
        df["team_one_most_consecutive_points"] - df["team_two_most_consecutive_points"]
    )
    df["g1_score_diff"] = df["g1_t1"] - df["g1_t2"]
    df["consec_g1_diff"] = (
        df["team_one_most_consecutive_points_game_1"]
        - df["team_two_most_consecutive_points_game_1"]
    )
    df["game_pts_g1_diff"] = (
        df["team_one_game_points_game_1"] - df["team_two_game_points_game_1"]
    )
    df["g1_total"] = df["g1_t1"] + df["g1_t2"]
    df["t1_win_pct_g1"] = df["g1_t1"] / df["g1_total"].replace(0, np.nan)
    df["round_num"] = df["round"].map(ROUND_ORDER)
    df["tournament_tier"] = df["tournament_type"].map(TIER_ORDER)

    engineered_df = df[FEATURES + [TARGET]].dropna()
    display_df = df.loc[engineered_df.index].reset_index(drop=True)
    model_df = engineered_df.reset_index(drop=True)
    return model_df, display_df


@st.cache_resource(show_spinner=False)
def train_models(model_df):
    X = model_df[FEATURES]
    y = model_df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": (
            LogisticRegression(max_iter=1000, random_state=42),
            X_train_scaled,
            X_test_scaled,
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=150, random_state=42),
            X_train,
            X_test,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            X_train,
            X_test,
        ),
    }

    rows = []
    fitted = {}
    for name, (model, train_matrix, test_matrix) in models.items():
        model.fit(train_matrix, y_train)
        probabilities = model.predict_proba(test_matrix)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, predictions),
                "f1": f1_score(y_test, predictions),
                "roc_auc": roc_auc_score(y_test, probabilities),
            }
        )
        fitted[name] = model

    metrics = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    best_name = metrics.iloc[0]["model"]
    return fitted, metrics, best_name, scaler


def positive_class_values(values):
    if isinstance(values, list):
        return values[1] if len(values) > 1 else values[0]
    arr = np.asarray(values)
    if arr.ndim == 3:
        return arr[:, :, 1] if arr.shape[2] > 1 else arr[:, :, 0]
    return arr


def positive_class_expected_value(value):
    if isinstance(value, list):
        return value[1] if len(value) > 1 else value[0]
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    flat = arr.ravel()
    return flat[1] if flat.size > 1 else flat[0]


def prediction_input(row, model_name, scaler):
    values = row[FEATURES].to_frame().T
    if model_name == "Logistic Regression":
        return scaler.transform(values)
    return values


def predict_probability(model, row, model_name, scaler):
    return float(model.predict_proba(prediction_input(row, model_name, scaler))[0, 1])


def build_lime_explanation(model, training_frame, selected_features):
    explainer = LimeTabularExplainer(
        training_data=training_frame[FEATURES].to_numpy(),
        feature_names=FEATURES,
        class_names=["Team two wins", "Team one wins"],
        mode="classification",
        random_state=42,
    )
    return explainer.explain_instance(
        data_row=selected_features[FEATURES].to_numpy(),
        predict_fn=lambda rows: model.predict_proba(pd.DataFrame(rows, columns=FEATURES)),
        num_features=10,
    )


def confidence_label(probability):
    distance = abs(probability - 0.5)
    if distance >= 0.35:
        return "High confidence", "The model sees a clear performance gap."
    if distance >= 0.18:
        return "Moderate confidence", "The model sees an advantage, but not a runaway case."
    return "Borderline", "The model sees a close match where small factors may matter."


def make_manual_row(model_df):
    medians = model_df[FEATURES].median()
    st.subheader("Prediction Form")
    st.write("Enter match-performance and context values to estimate the probability of team one winning.")

    col_a, col_b = st.columns(2)
    with col_a:
        team_one_total = st.number_input(
            "Team one total points",
            min_value=0,
            max_value=100,
            value=int(round(medians["team_one_total_points"])),
        )
        team_one_game_points = st.number_input(
            "Team one game points",
            min_value=0,
            max_value=10,
            value=int(round(medians["team_one_game_points"])),
        )
        team_one_consecutive = st.number_input(
            "Team one most consecutive points",
            min_value=0,
            max_value=25,
            value=int(round(medians["team_one_most_consecutive_points"])),
        )
        team_one_g1_consecutive = st.number_input(
            "Team one consecutive points in game 1",
            min_value=0,
            max_value=25,
            value=int(round(medians["team_one_most_consecutive_points"])),
        )
        team_one_g1_game_points = st.number_input(
            "Team one game points in game 1",
            min_value=0,
            max_value=10,
            value=int(round(medians["team_one_game_points"])),
        )
    with col_b:
        team_two_total = st.number_input(
            "Team two total points",
            min_value=0,
            max_value=100,
            value=int(round(medians["team_two_total_points"])),
        )
        team_two_game_points = st.number_input(
            "Team two game points",
            min_value=0,
            max_value=10,
            value=int(round(medians["team_two_game_points"])),
        )
        team_two_consecutive = st.number_input(
            "Team two most consecutive points",
            min_value=0,
            max_value=25,
            value=int(round(medians["team_two_most_consecutive_points"])),
        )
        team_two_g1_consecutive = st.number_input(
            "Team two consecutive points in game 1",
            min_value=0,
            max_value=25,
            value=int(round(medians["team_two_most_consecutive_points"])),
        )
        team_two_g1_game_points = st.number_input(
            "Team two game points in game 1",
            min_value=0,
            max_value=10,
            value=int(round(medians["team_two_game_points"])),
        )

    col_c, col_d, col_e, col_f = st.columns(4)
    with col_c:
        g1_t1 = st.slider("Game 1 score: team one", 0, 30, 21)
    with col_d:
        g1_t2 = st.slider("Game 1 score: team two", 0, 30, 18)
    with col_e:
        nb_sets = st.selectbox("Number of sets", [2, 3], index=0)
    with col_f:
        round_name = st.selectbox("Round", list(ROUND_ORDER.keys()), index=7)

    tier_name = st.selectbox(
        "Tournament tier",
        list(TIER_ORDER.keys()),
        index=2,
    )

    g1_total = max(g1_t1 + g1_t2, 1)
    row = pd.Series(
        {
            "total_points_diff": team_one_total - team_two_total,
            "game_points_diff": team_one_game_points - team_two_game_points,
            "consec_points_diff": team_one_consecutive - team_two_consecutive,
            "g1_score_diff": g1_t1 - g1_t2,
            "consec_g1_diff": team_one_g1_consecutive - team_two_g1_consecutive,
            "game_pts_g1_diff": team_one_g1_game_points - team_two_g1_game_points,
            "team_one_total_points": team_one_total,
            "team_two_total_points": team_two_total,
            "team_one_game_points": team_one_game_points,
            "team_two_game_points": team_two_game_points,
            "team_one_most_consecutive_points": team_one_consecutive,
            "team_two_most_consecutive_points": team_two_consecutive,
            "g1_total": g1_total,
            "t1_win_pct_g1": g1_t1 / g1_total,
            "round_num": ROUND_ORDER[round_name],
            "tournament_tier": TIER_ORDER[tier_name],
            "nb_sets": nb_sets,
        }
    )
    return row, round_name, tier_name


def render_result(probability, model_name):
    predicted_winner = "Team one" if probability >= 0.5 else "Team two"
    level, message = confidence_label(probability)
    css_class = "result-win" if probability >= 0.5 else "result-loss"
    st.markdown(
        f"""
        <div class="{css_class}">
            <h3>{predicted_winner} predicted</h3>
            <p><strong>Team one win probability:</strong> {probability:.1%}</p>
            <p><strong>{level}:</strong> {message}</p>
            <p><strong>Prediction model:</strong> {model_name}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def shap_summary(model, data):
    sample = data[FEATURES].sample(n=min(300, len(data)), random_state=42)
    explainer = shap.TreeExplainer(model)
    values = positive_class_values(explainer.shap_values(sample))
    expected = positive_class_expected_value(explainer.expected_value)
    summary = pd.DataFrame(
        {"feature": FEATURES, "mean_abs_shap": np.abs(values).mean(axis=0)}
    ).sort_values("mean_abs_shap", ascending=False)
    return explainer, expected, summary


def render_local_explanations(xai_model, model_df, selected_features):
    explainer = shap.TreeExplainer(xai_model)
    selected_frame = selected_features[FEATURES].to_frame().T
    local_values = positive_class_values(explainer.shap_values(selected_frame))[0]
    expected = positive_class_expected_value(explainer.expected_value)
    local_df = pd.DataFrame(
        {
            "feature": FEATURES,
            "value": selected_frame.iloc[0].values,
            "shap_effect": local_values,
        }
    )
    local_df["direction"] = np.where(
        local_df["shap_effect"] >= 0,
        "pushes toward team one",
        "pushes toward team two",
    )
    local_df = local_df.reindex(
        local_df["shap_effect"].abs().sort_values(ascending=False).index
    )

    st.write("Top SHAP drivers for this prediction")
    st.dataframe(
        local_df.head(10).style.format({"value": "{:.3f}", "shap_effect": "{:.3f}"}),
        width="stretch",
    )

    explanation = shap.Explanation(
        values=local_values,
        base_values=expected,
        data=selected_frame.iloc[0].values,
        feature_names=FEATURES,
    )
    fig, _ = plt.subplots(figsize=(9, 5))
    shap.plots.waterfall(explanation, max_display=10, show=False)
    st.pyplot(fig, clear_figure=True)

    lime_explanation = build_lime_explanation(xai_model, model_df, selected_features)
    lime_rows = pd.DataFrame(
        lime_explanation.as_list(),
        columns=["condition", "local_weight"],
    )
    lime_rows["direction"] = np.where(
        lime_rows["local_weight"] >= 0,
        "supports team one",
        "supports team two",
    )
    st.write("LIME local explanation")
    st.dataframe(
        lime_rows.style.format({"local_weight": "{:.4f}"}),
        width="stretch",
        hide_index=True,
    )
    fig = lime_explanation.as_pyplot_figure()
    st.pyplot(fig, clear_figure=True)


raw_df = load_local_data()
uploaded_file = st.sidebar.file_uploader("Upload ms.csv", type=["csv"])
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

st.sidebar.title("Badminton XAI")
page = st.sidebar.radio(
    "Navigation",
    ["Predict", "Dashboard", "Explainability", "Dataset", "About"],
)

if raw_df is None:
    st.markdown(
        """
        <div class="hero">
            <h1>Badminton XAI Match Prediction</h1>
            <p>Upload <code>ms.csv</code> in the sidebar to train the demo model and use the system.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

required_columns = {
    "winner",
    "game_1_score",
    "round",
    "tournament_type",
    "team_one_players",
    "team_two_players",
}
missing_columns = required_columns.difference(raw_df.columns)
if missing_columns:
    st.error(f"The dataset is missing required columns: {sorted(missing_columns)}")
    st.stop()

with st.spinner("Training models and preparing dashboard..."):
    model_df, display_df = prepare_data(raw_df)
    fitted_models, metrics_df, best_model_name, scaler = train_models(model_df)

best_model = fitted_models[best_model_name]
xai_model_name = "Random Forest"
xai_model = fitted_models[xai_model_name]

st.markdown(
    """
    <div class="hero">
        <h1>Explainable Machine Learning for Badminton Match Prediction</h1>
        <p>Decision-support system for binary match outcome prediction, performance analytics, and transparent SHAP/LIME explanations.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

summary_cols = st.columns(4)
summary_cols[0].metric("Clean matches", f"{len(model_df):,}")
summary_cols[1].metric("Input features", len(FEATURES))
summary_cols[2].metric("Best model", best_model_name)
summary_cols[3].metric("Best ROC-AUC", f"{metrics_df.iloc[0]['roc_auc']:.4f}")

if page == "Predict":
    st.header("Predict Match Outcome")
    mode = st.segmented_control(
        "Prediction mode",
        ["Manual form", "Historical match"],
        default="Manual form",
    )

    if mode == "Manual form":
        selected_features, round_name, tier_name = make_manual_row(model_df)
        probability = predict_probability(best_model, selected_features, best_model_name, scaler)
        render_result(probability, best_model_name)
        with st.expander("Input summary", expanded=True):
            summary = selected_features.to_frame("value").reset_index()
            summary.columns = ["feature", "value"]
            st.dataframe(summary, width="stretch", hide_index=True)
            st.caption(f"Round: {round_name} | Tournament tier: {tier_name}")
    else:
        match_index = st.slider("Select historical match row", 0, len(display_df) - 1, 0)
        selected_row = display_df.iloc[match_index]
        selected_features = model_df.iloc[match_index]
        probability = predict_probability(best_model, selected_features, best_model_name, scaler)
        render_result(probability, best_model_name)
        actual = "Team one" if selected_row["winner"] == 1 else "Team two"
        st.metric("Actual winner", actual)
        st.dataframe(
            pd.DataFrame(
                {
                    "team": ["Team one", "Team two"],
                    "player": [
                        selected_row.get("team_one_players", "Unknown"),
                        selected_row.get("team_two_players", "Unknown"),
                    ],
                    "total_points": [
                        selected_row.get("team_one_total_points", np.nan),
                        selected_row.get("team_two_total_points", np.nan),
                    ],
                    "game_points": [
                        selected_row.get("team_one_game_points", np.nan),
                        selected_row.get("team_two_game_points", np.nan),
                    ],
                }
            ),
            width="stretch",
            hide_index=True,
        )

    st.subheader("Why this prediction?")
    st.caption(f"XAI model: {xai_model_name}. This keeps explanations stable and directly linked to the project methodology.")
    render_local_explanations(xai_model, model_df, selected_features)

elif page == "Dashboard":
    st.header("Analytics Dashboard")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.metric("Team one wins", f"{int(model_df[TARGET].sum()):,}")
    col_b.metric("Team two wins", f"{int((1 - model_df[TARGET]).sum()):,}")
    col_c.metric("Average total points diff", f"{model_df['total_points_diff'].mean():.2f}")
    col_d.metric("Average game 1 total", f"{model_df['g1_total'].mean():.1f}")

    st.subheader("Model Performance")
    st.dataframe(
        metrics_df.set_index("model").style.format(
            {"accuracy": "{:.4f}", "f1": "{:.4f}", "roc_auc": "{:.4f}"}
        ),
        width="stretch",
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Outcome Distribution")
        outcomes = model_df[TARGET].map({1: "Team one wins", 0: "Team two wins"}).value_counts()
        st.bar_chart(outcomes)

        st.subheader("Tournament Tier Distribution")
        tier_counts = display_df["tournament_type"].value_counts().head(8)
        st.bar_chart(tier_counts)
    with right:
        st.subheader("Round Distribution")
        round_counts = display_df["round"].value_counts().head(10)
        st.bar_chart(round_counts)

        st.subheader("Feature Correlation with Outcome")
        corr = model_df[FEATURES + [TARGET]].corr(numeric_only=True)[TARGET].drop(TARGET)
        corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(12)
        st.bar_chart(corr)

elif page == "Explainability":
    st.header("SHAP and LIME Explainability")
    st.markdown(
        """
        <div class="info-band">
        SHAP explains global feature influence and single-match contributions.
        LIME builds a simpler local model around one selected match so the decision can be interpreted case by case.
        </div>
        """,
        unsafe_allow_html=True,
    )

    explainer, expected, global_values = shap_summary(xai_model, model_df)
    st.subheader("Global SHAP Importance")
    st.bar_chart(global_values.set_index("feature")["mean_abs_shap"])
    st.dataframe(global_values.head(12), width="stretch", hide_index=True)

    st.subheader("Local Match Explanation")
    match_index = st.slider("Select match for XAI explanation", 0, len(display_df) - 1, 0)
    selected_row = display_df.iloc[match_index]
    selected_features = model_df.iloc[match_index]
    probability = predict_probability(best_model, selected_features, best_model_name, scaler)
    render_result(probability, best_model_name)
    st.caption(
        f"{selected_row.get('team_one_players', 'Team one')} vs "
        f"{selected_row.get('team_two_players', 'Team two')}"
    )
    render_local_explanations(xai_model, model_df, selected_features)

elif page == "Dataset":
    st.header("Dataset Explorer")
    st.write("Historical BWF World Tour men's singles match data used for this demonstration.")
    st.dataframe(raw_df.head(100), width="stretch")

    st.subheader("Columns and Missing Values")
    missing = pd.DataFrame(
        {
            "column": raw_df.columns,
            "dtype": [str(raw_df[col].dtype) for col in raw_df.columns],
            "missing": [int(raw_df[col].isna().sum()) for col in raw_df.columns],
            "missing_pct": [round(raw_df[col].isna().mean() * 100, 2) for col in raw_df.columns],
        }
    ).sort_values("missing_pct", ascending=False)
    st.dataframe(missing, width="stretch", hide_index=True)

elif page == "About":
    st.header("About the Project")
    st.markdown(
        """
        This system is aligned with the MSc project specification:
        **Explainable Machine Learning for Badminton Match Prediction and Performance Analytics**.

        **Objectives represented in the app**
        - Load and preprocess public badminton match data.
        - Engineer performance and contextual features.
        - Train Logistic Regression, Random Forest, and Gradient Boosting classifiers.
        - Compare models using accuracy, F1-score, and ROC-AUC.
        - Use SHAP and LIME to interpret global and local model behaviour.
        - Present an interactive dashboard and prediction interface.

        **Methodology**
        1. Historical BWF match data is cleaned and filtered to complete win/loss outcomes.
        2. Game scores and match statistics are transformed into differential and contextual features.
        3. Supervised classifiers are trained on a stratified train/test split.
        4. The best model is used for prediction, while Random Forest is used for stable SHAP/LIME explanations.
        5. The system reports both prediction probability and interpretable feature contributions.

        **Important limitation**
        The current feature set is mainly performance-based. It is strongest for performance analytics and
        post-match or in-match explanation. For a purely pre-match prediction system, ranking, recent form,
        head-to-head history, and player-level features should be added.
        """
    )
