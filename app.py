import io
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score, roc_curve)

BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"
BUNDLED_TEST_CSV = BASE_DIR / "test_data.csv"

AUTHOR = "Vadlamudi Jurendra Veeraiah Chowdary"
STUDENT_ID = "2025AC05994"

METRIC_ORDER = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]

METRIC_HELP = {
    "Accuracy": "Share of all clients classified correctly. Misleading here — "
                "predicting “no” for everyone already scores 88.3%.",
    "AUC": "Ranking quality across every threshold. Unaffected by where the cut-off sits.",
    "Precision": "Of the clients we call, how many actually subscribe.",
    "Recall": "Of the clients who would subscribe, how many we manage to reach.",
    "F1": "Harmonic mean of precision and recall.",
    "MCC": "Correlation between prediction and truth using all four matrix cells. "
           "The most trustworthy single number under class imbalance.",
}

st.set_page_config(
    page_title="Term Deposit Campaign Console",
    page_icon="📞",
    layout="wide",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1350px; }
      .meter-grid { display: flex; gap: 0.6rem; flex-wrap: wrap; }
      .meter {
          flex: 1 1 150px; padding: 0.7rem 0.85rem; border-radius: 0.6rem;
          background: rgba(128, 128, 128, 0.10);
          border: 1px solid rgba(128, 128, 128, 0.22);
      }
      .meter .label {
          font-size: 0.74rem; text-transform: uppercase; letter-spacing: 0.06em;
          opacity: 0.72; font-weight: 600;
      }
      .meter .value { font-size: 1.65rem; font-weight: 700; line-height: 1.25; }
      .meter .bar {
          height: 5px; border-radius: 3px; margin-top: 0.35rem;
          background: rgba(128, 128, 128, 0.25);
      }
      .meter .bar > span { display: block; height: 100%; border-radius: 3px; }
      .meter .foot { font-size: 0.72rem; opacity: 0.62; margin-top: 0.3rem; }
      .outcome {
          padding: 0.8rem 0.95rem; border-radius: 0.6rem; height: 100%;
          background: rgba(128, 128, 128, 0.08);
          border-left: 4px solid var(--accent);
      }
      .outcome .n { font-size: 1.5rem; font-weight: 700; }
      .outcome .t { font-size: 0.8rem; font-weight: 600; opacity: 0.85; }
      .outcome .d { font-size: 0.74rem; opacity: 0.62; margin-top: 0.25rem; }
      .rule { height: 1px; background: rgba(128,128,128,0.25); margin: 1.6rem 0 1.1rem; }
      .byline {
          display: flex; flex-wrap: wrap; align-items: center; gap: 0.55rem;
          padding: 0.5rem 0.85rem; margin-bottom: 0.9rem; border-radius: 0.5rem;
          background: rgba(128, 128, 128, 0.10);
          border-left: 4px solid #2e6fdb;
      }
      .byline .who { font-weight: 700; font-size: 0.95rem; }
      .byline .id {
          font-family: ui-monospace, "Cascadia Code", Consolas, monospace;
          font-size: 0.82rem; font-weight: 600; padding: 0.1rem 0.45rem;
          border-radius: 0.3rem; background: rgba(46, 111, 219, 0.18);
      }
      .footer {
          margin-top: 2.2rem; padding-top: 1rem; font-size: 0.8rem; opacity: 0.7;
          border-top: 1px solid rgba(128, 128, 128, 0.25); text-align: center;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_json(name):
    return json.loads((MODEL_DIR / name).read_text())


@st.cache_resource(show_spinner="Loading trained pipelines…")
def load_pipelines(file_stems):
    return {name: joblib.load(MODEL_DIR / f"{stem}.pkl") for name, stem in file_stems.items()}


@st.cache_data(show_spinner=False)
def read_csv(payload):
    return pd.read_csv(io.BytesIO(payload))


SCHEMA = load_json("schema.json")
REFERENCE = load_json("metrics.json")
PIPELINES = load_pipelines(SCHEMA["files"])
TARGET_COL = SCHEMA["target"]
POSITIVE_LABEL = SCHEMA["positive_label"]
REQUIRED_FEATURES = SCHEMA["all_features"]


def prepare(frame):
    notes = []
    frame = frame.copy()
    frame.columns = [c.strip() for c in frame.columns]

    if "duration" in frame.columns:
        frame = frame.drop(columns=["duration"])
        notes.append("Dropped `duration` — excluded during training as target leakage.")

    y = None
    if TARGET_COL in frame.columns:
        raw = frame.pop(TARGET_COL)
        if pd.api.types.is_numeric_dtype(raw):
            y = (raw.fillna(0).astype(float) > 0).astype(int)
        else:
            text = raw.astype(str).str.strip().str.lower()
            positive = {POSITIVE_LABEL, "1", "true", "t", "y"}
            negative = {"no", "0", "false", "f", "n"}
            unexpected = sorted(set(text) - positive - negative)
            if unexpected:
                raise ValueError(
                    f"Could not read the `{TARGET_COL}` column. Unexpected value(s): "
                    f"{', '.join(repr(v) for v in unexpected[:5])}. "
                    f"Expected `{POSITIVE_LABEL}`/`no` or `1`/`0`."
                )
            y = text.isin(positive).astype(int)
    else:
        notes.append(
            f"No `{TARGET_COL}` column — predictions will be shown, but metrics "
            "cannot be computed without ground-truth labels."
        )

    missing = [c for c in REQUIRED_FEATURES if c not in frame.columns]
    if missing:
        raise ValueError(f"Uploaded CSV is missing required column(s): {', '.join(missing)}")

    extra = [c for c in frame.columns if c not in REQUIRED_FEATURES]
    if extra:
        frame = frame.drop(columns=extra)
        notes.append(f"Ignored {len(extra)} unrecognised column(s): {', '.join(extra)}.")

    return frame[REQUIRED_FEATURES], y, notes


def score_at(proba, y_true, threshold):
    pred = (proba >= threshold).astype(int)
    return {
        "Accuracy": accuracy_score(y_true, pred),
        "AUC": roc_auc_score(y_true, proba),
        "Precision": precision_score(y_true, pred, zero_division=0),
        "Recall": recall_score(y_true, pred, zero_division=0),
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
    }


@st.cache_data(show_spinner=False)
def probabilities(model_name, payload, _pipeline, _X):
    return _pipeline.predict_proba(_X)[:, 1]


def meter_html(label, value, reference=None):
    hue = 12 + 118 * max(0.0, min(1.0, value))       # red → green
    foot = "" if reference is None else (
        f"<div class='foot'>{value - reference:+.4f} vs notebook</div>"
    )
    return (
        f"<div class='meter' title='{METRIC_HELP.get(label, '')}'>"
        f"<div class='label'>{label}</div>"
        f"<div class='value'>{value:.4f}</div>"
        f"<div class='bar'><span style='width:{max(2, value * 100):.0f}%;"
        f"background:hsl({hue},62%,48%)'></span></div>"
        f"{foot}</div>"
    )


def outcome_html(count, title, description, accent):
    return (
        f"<div class='outcome' style='--accent:{accent}'>"
        f"<div class='n'>{count:,}</div>"
        f"<div class='t'>{title}</div>"
        f"<div class='d'>{description}</div></div>"
    )


def section(title, subtitle=""):
    st.markdown("<div class='rule'></div>", unsafe_allow_html=True)
    st.subheader(title)
    if subtitle:
        st.caption(subtitle)


st.title("📞 Term Deposit Campaign Console")
st.markdown(
    f"<div class='byline'>"
    f"<span class='who'>{AUTHOR}</span>"
    f"<span class='id'>{STUDENT_ID}</span>"
    f"</div>",
    unsafe_allow_html=True,
)
st.markdown(
    "A Portuguese bank calls clients to sell term deposits, but only about **one in nine** "
    "subscribes. These five classifiers rank the call list so agents reach the likely "
    "subscribers first. Trained on the "
    "[UCI Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing) dataset."
)

# ----------------------------------------------------------------------------- controls
with st.container(border=True):
    st.markdown("##### 📤 Upload test data (CSV)")
    uploaded = st.file_uploader(
        "Needs the 15 predictor columns. Include a `y` column (`yes`/`no` or `1`/`0`) "
        "to compute evaluation metrics.",
        type="csv",
        help="Leave empty to use the bundled test_data.csv — the 9,043-row held-out "
             "split created in the training notebook.",
    )

    col_model, col_threshold = st.columns([1, 1.5])

    with col_model:
        model_name = st.selectbox(
            "🤖 Classifier",
            list(PIPELINES),
            index=len(PIPELINES) - 1,
            help="All five are scored in the leaderboard below; this selects the one to inspect.",
        )

    with col_threshold:
        threshold = st.slider(
            "🎚️ Decision threshold — probability above which we place a call",
            0.05, 0.95, 0.50, 0.05,
            help="0.50 is the scikit-learn default and reproduces the README figures. "
                 "Lower it to call more people and catch more subscribers.",
        )

if uploaded is not None:
    payload = uploaded.getvalue()
    origin = f"your uploaded file **{uploaded.name}**"
elif BUNDLED_TEST_CSV.exists():
    payload = BUNDLED_TEST_CSV.read_bytes()
    origin = "the bundled **test_data.csv** (upload a file above to score your own)"
else:
    st.error("`test_data.csv` is missing. Upload a CSV above to continue.")
    st.stop()

try:
    X, y_true, notes = prepare(read_csv(payload))
except ValueError as err:
    st.error(str(err))
    if "missing required column" in str(err).lower():
        st.info(f"Expected columns: `{'`, `'.join(REQUIRED_FEATURES)}` "
                f"(plus optional `{TARGET_COL}`).")
    st.stop()
except Exception as err:
    st.error(f"Could not read that file: {err}")
    st.stop()

for note in notes:
    st.info(note, icon="ℹ️")

if y_true is not None and y_true.nunique() < 2:
    st.warning("The test data contains only one class, so AUC and MCC are undefined.", icon="⚠️")
    y_true = None

if uploaded is not None:
    st.success(f"Scoring {origin} — {len(X):,} clients.", icon="✅")

summary = f"Scoring {origin} · **{len(X):,}** clients · **{len(X.columns)}** predictors"
if y_true is not None:
    summary += (f" · **{int(y_true.sum()):,}** actually subscribed "
                f"({y_true.mean():.1%} — the base rate to beat)")
st.caption(summary)

if y_true is None:
    st.warning(
        "Without a `y` column only predictions can be shown. Add ground-truth labels to "
        "compute evaluation metrics.",
        icon="⚠️",
    )

scores = {name: probabilities(name, payload, pipe, X) for name, pipe in PIPELINES.items()}
proba = scores[model_name]
pred = (proba >= threshold).astype(int)


section(
    "Model leaderboard",
    f"All five classifiers scored on this data at a {threshold:.2f} threshold, "
    "ranked by MCC.",
)

if y_true is None:
    counts = pd.DataFrame({
        "Model": list(PIPELINES),
        "Clients to call": [int((p >= threshold).sum()) for p in scores.values()],
    }).set_index("Model")
    st.dataframe(counts, width="stretch")
else:
    board = pd.DataFrame(
        {name: score_at(p, y_true, threshold) for name, p in scores.items()}
    ).T[METRIC_ORDER]
    board = board.sort_values("MCC", ascending=False)
    board.index.name = "Model"

    st.dataframe(
        board,
        width="stretch",
        column_config={
            metric: st.column_config.ProgressColumn(
                metric, min_value=0.0, max_value=1.0, format="%.4f", help=METRIC_HELP[metric]
            )
            for metric in METRIC_ORDER
        },
    )

    winner = board.index[0]
    runner_up = board.index[1]
    top_accuracy = board["Accuracy"].idxmax()
    verdict = (
        f"**{winner}** leads on MCC ({board.loc[winner, 'MCC']:.4f}), ahead of "
        f"{runner_up} ({board.loc[runner_up, 'MCC']:.4f})."
    )
    if top_accuracy != winner:
        verdict += (
            f" Note that **{top_accuracy}** has the highest *accuracy* "
            f"({board.loc[top_accuracy, 'Accuracy']:.4f}) yet recall of only "
            f"{board.loc[top_accuracy, 'Recall']:.4f} — it scores well by mostly "
            "predicting “no”, which is why MCC is the metric to trust here."
        )
    st.success(verdict, icon="🏁")


section(f"Inspecting · {model_name}", "Evaluation metrics on the data loaded above.")

if y_true is not None:
    metrics = score_at(proba, y_true, threshold)
    reference = REFERENCE["models"].get(model_name, {})
    at_default = abs(threshold - 0.50) < 1e-9

    st.markdown(
        "<div class='meter-grid'>"
        + "".join(
            meter_html(m, metrics[m], reference.get(m) if at_default else None)
            for m in METRIC_ORDER
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.caption(
        "Compared against the training notebook's reference figures."
        if at_default else
        f"Threshold moved to {threshold:.2f} — figures differ from the README, which uses 0.50."
    )

    # ------------------------------------------------------------------- call outcomes
    section("Call outcomes", "The confusion matrix, read as campaign results.")
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    cards = st.columns(4)
    cards[0].markdown(
        outcome_html(tp, "Subscribers reached", "Called and did subscribe — the revenue.",
                     "#2e9e5b"), unsafe_allow_html=True)
    cards[1].markdown(
        outcome_html(fn, "Subscribers missed", "Would have subscribed, never called. "
                     "The costly error.", "#c0392b"), unsafe_allow_html=True)
    cards[2].markdown(
        outcome_html(fp, "Wasted calls", "Called but declined — a few minutes of agent time.",
                     "#d98c1f"), unsafe_allow_html=True)
    cards[3].markdown(
        outcome_html(tn, "Correctly skipped", "Not called, would have declined anyway.",
                     "#7f8c8d"), unsafe_allow_html=True)

    if tp:
        st.caption(
            f"**{tp + fp:,}** calls to reach **{tp:,}** subscribers — "
            f"**{(tp + fp) / tp:.1f}** calls per conversion, capturing "
            f"**{tp / (tp + fn):.1%}** of all available subscribers."
        )

    matrix_col, report_col = st.columns([1, 1.25])

    with matrix_col:
        fig, ax = plt.subplots(figsize=(4.1, 3.5))
        matrix = np.array([[tn, fp], [fn, tp]])
        ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{matrix[i, j]:,}", ha="center", va="center", fontsize=13,
                        color="white" if matrix[i, j] > matrix.max() * 0.55 else "#1b2a41")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["no", "yes"],
               yticklabels=["no", "yes"], xlabel="Predicted", ylabel="Actual",
               title="Confusion matrix")
        fig.tight_layout()
        st.pyplot(fig)

    with report_col:
        st.markdown("**Classification report**")
        report = classification_report(
            y_true, pred, target_names=["no (0)", "yes (1)"],
            output_dict=True, zero_division=0,
        )
        st.dataframe(pd.DataFrame(report).T.round(4), width="stretch")

    section("Threshold behaviour", "How the choice of cut-off changes what the model does.")
    roc_col, sweep_col = st.columns(2)

    with roc_col:
        fig, ax = plt.subplots(figsize=(5.1, 4))
        for name, p in scores.items():
            fpr, tpr, _ = roc_curve(y_true, p)
            ax.plot(fpr, tpr, lw=2.2 if name == model_name else 1.1,
                    alpha=1.0 if name == model_name else 0.45,
                    label=f"{name} ({roc_auc_score(y_true, p):.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Chance (0.500)")
        ax.set(xlabel="False positive rate", ylabel="True positive rate",
               title="ROC — all models, selected highlighted")
        ax.legend(loc="lower right", fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)

    with sweep_col:
        grid = np.arange(0.05, 0.96, 0.05)
        sweep = pd.DataFrame(
            [score_at(proba, y_true, t) for t in grid], index=grid.round(2)
        )[["Precision", "Recall", "F1", "MCC"]]

        fig, ax = plt.subplots(figsize=(5.1, 4))
        for column in sweep.columns:
            ax.plot(sweep.index, sweep[column], marker="o", ms=3, lw=1.4, label=column)
        ax.axvline(threshold, color="crimson", ls="--", lw=1.2,
                   label=f"current ({threshold:.2f})")
        ax.set(xlabel="Decision threshold", ylabel="Score",
               title=f"{model_name} — metrics vs threshold")
        ax.legend(fontsize=7.5)
        fig.tight_layout()
        st.pyplot(fig)

        best_threshold = sweep["MCC"].idxmax()
        st.caption(
            f"MCC peaks at **{sweep['MCC'].max():.4f}** with a threshold of "
            f"**{best_threshold:.2f}** for this model."
        )

section("Prioritised call list",
        "Every client ranked by predicted probability — the output an agent would work through.")

call_list = X.copy()
call_list.insert(0, "call?", np.where(pred == 1, "✅ call", "—"))
call_list.insert(1, "probability", proba.round(4))
if y_true is not None:
    call_list.insert(2, "actual", np.where(y_true == 1, "yes", "no"))
    call_list.insert(3, "correct", np.where(pred == y_true, "✓", "✗"))

filter_col, count_col = st.columns([2, 1])
with filter_col:
    view = st.radio(
        "Show", ["Everyone", "Only those we would call", "Only mistakes"],
        horizontal=True, label_visibility="collapsed", disabled=y_true is None,
    )

shown = call_list
if view == "Only those we would call":
    shown = call_list[call_list["call?"] == "✅ call"]
elif view == "Only mistakes" and y_true is not None:
    shown = call_list[call_list["correct"] == "✗"]

count_col.caption(f"{len(shown):,} of {len(call_list):,} clients · top 500 shown")
st.dataframe(
    shown.sort_values("probability", ascending=False).head(500),
    width="stretch", height=420,
    column_config={
        "probability": st.column_config.ProgressColumn(
            "probability", min_value=0.0, max_value=1.0, format="%.4f",
            help="Model's estimated chance this client subscribes.",
        )
    },
)

buffer = io.StringIO()
call_list.to_csv(buffer, index=False)
st.download_button(
    "⬇️ Download the full ranked list (CSV)",
    buffer.getvalue(),
    file_name=f"call_list_{SCHEMA['files'][model_name]}.csv",
    mime="text/csv",
)


st.markdown(
    f"<div class='footer'>"
    f"<strong>{AUTHOR}</strong> &nbsp;·&nbsp; {STUDENT_ID}"
    f"</div>",
    unsafe_allow_html=True,
)
