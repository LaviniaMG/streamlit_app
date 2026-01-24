import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# =========================
# ML imports
# =========================
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)

# =========================
# App config
# =========================
st.set_page_config(
    page_title="EDA & ML App",
    layout="wide",
    initial_sidebar_state="expanded"
)

DEFAULT_FILE = "titanic.csv"

# =========================
# Sidebar
# =========================
st.sidebar.title("Navigare")
menu = [
    "Upload & Filtrare",
    "Structura Dataset",
    "Analiza Numerica",
    "Analiza Categorica",
    "Corelatii si Outlieri",
    "ML – Train & Compare"
]
page = st.sidebar.radio("Selecteaza pagina:", menu)

# =========================
# Upload & Filtrare
# =========================
if page == "Upload & Filtrare":
    st.title("EDA – Dataset")

    file = st.file_uploader("Incarca fisier CSV sau Excel", type=["csv", "xlsx"])

    if "df" not in st.session_state:
        st.session_state.df = None

    if file is not None:
        if file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(file)
        else:
            st.session_state.df = pd.read_excel(file)
        st.success("Fisier incarcat cu succes!")
    elif os.path.exists(DEFAULT_FILE) and st.session_state.df is None:
        st.session_state.df = pd.read_csv(DEFAULT_FILE)
        st.info("Se foloseste fisierul default")

    if st.session_state.df is not None:
        df = st.session_state.df
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        st.subheader("Primele 10 randuri")
        st.dataframe(df.head(10))

# =========================
# Structura Dataset
# =========================
elif page == "Structura Dataset":
    df = st.session_state.get("df")
    if df is None:
        st.warning("Incarca datele mai intai.")
        st.stop()

    st.metric("Randuri", df.shape[0])
    st.metric("Coloane", df.shape[1])

    st.subheader("Tipuri de date")
    st.dataframe(pd.DataFrame({"Coloana": df.columns, "Tip": df.dtypes}))

    st.subheader("Valori lipsa (%)")
    missing_pct = df.isnull().mean() * 100
    st.dataframe(missing_pct.reset_index(name="Procent (%)"))

# =========================
# ML – TRAIN & COMPARE
# =========================
elif page == "ML – Train & Compare":

    df = st.session_state.get("df")
    if df is None:
        st.warning("Incarca datele mai intai.")
        st.stop()

    st.header("Machine Learning – Train, Evaluate & Compare")

    # =========================
    # Partea 1 – Problem Setup
    # =========================
    st.subheader("Problem Setup")

    target = st.selectbox("Coloana target", df.columns)

    select_all = st.checkbox("Select all features", value=True)
    if select_all:
        features = [c for c in df.columns if c != target]
    else:
        features = st.multiselect(
            "Selecteaza feature-urile",
            [c for c in df.columns if c != target]
        )

    problem_type = st.radio("Tip problema", ["Clasificare", "Regresie"])

    # =========================
    # Partea 2 – Preprocesare
    # =========================
    st.subheader("Preprocesare")

    imput_method = st.selectbox(
        "Metoda imputare numeric",
        ["mean", "median", "most_frequent"]
    )

    scaler_choice = st.selectbox(
        "Scalare",
        ["StandardScaler", "MinMaxScaler", "Fara"]
    )

    remove_outliers = st.checkbox("Eliminare outlieri (IQR)")
    use_feature_selection = st.checkbox("Selectie features (SelectKBest)")

    X = df[features]
    y = df[target]

    # Eliminare outlieri
    if remove_outliers:
        for col in X.select_dtypes(include=np.number).columns:
            Q1 = X[col].quantile(0.25)
            Q3 = X[col].quantile(0.75)
            IQR = Q3 - Q1
            mask = (X[col] >= Q1 - 1.5 * IQR) & (X[col] <= Q3 + 1.5 * IQR)
            X = X[mask]
            y = y[mask]

    num_cols = X.select_dtypes(include=np.number).columns
    cat_cols = X.select_dtypes(include="object").columns

    scaler = (
        StandardScaler() if scaler_choice == "StandardScaler"
        else MinMaxScaler() if scaler_choice == "MinMaxScaler"
        else "passthrough"
    )

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imput_method)),
        ("scaler", scaler)
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, num_cols),
        ("cat", categorical_pipeline, cat_cols)
    ])

    # =========================
    # Partea 3 – Split
    # =========================
    st.subheader("Train / Test Split")
    test_size = st.slider("Test size", 0.1, 0.5, 0.2)
    random_state = st.number_input("Random state", value=42)

    # =========================
    # Partea 4 – Models
    # =========================
    st.subheader("Models")

    model_names = st.multiselect(
        "Selecteaza modele",
        ["Logistic Regression", "Random Forest", "SVM"]
    )

    models = {}

    if problem_type == "Clasificare":
        if "Logistic Regression" in model_names:
            C = st.slider("LogReg C", 0.01, 10.0, 1.0)
            models["Logistic Regression"] = LogisticRegression(C=C, max_iter=1000)

        if "Random Forest" in model_names:
            n_estimators = st.slider("RF estimators", 50, 300, 100)
            max_depth = st.slider("RF max_depth", 2, 20, 5)
            models["Random Forest"] = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state
            )

        if "SVM" in model_names:
            C = st.slider("SVM C", 0.01, 10.0, 1.0)
            models["SVM"] = SVC(C=C, probability=True)

    else:
        if "Random Forest" in model_names:
            models["Random Forest"] = RandomForestRegressor(random_state=random_state)
        if "SVM" in model_names:
            models["SVM"] = SVR()
        if "Logistic Regression" in model_names:
            models["Linear Regression"] = LinearRegression()

    # =========================
    # Train & Evaluate
    # =========================
    if st.button("Train Models"):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        results = []

        for name, model in models.items():
            steps = [("prep", preprocessor)]

            if use_feature_selection:
                selector = SelectKBest(
                    f_classif if problem_type == "Clasificare" else f_regression,
                    k=min(10, X_train.shape[1])
                )
                steps.append(("select", selector))

            steps.append(("model", model))
            pipe = Pipeline(steps)

            pipe.fit(X_train, y_train)
            y_pred = pipe.predict(X_test)

            st.subheader(name)

            if problem_type == "Clasificare":
                acc = accuracy_score(y_test, y_pred)
                prec = precision_score(y_test, y_pred, average="weighted")
                rec = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                y_proba = pipe.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) == 2 else None
                roc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

                results.append({
                    "Model": name,
                    "Accuracy": acc,
                    "Precision": prec,
                    "Recall": rec,
                    "F1": f1,
                    "ROC-AUC": roc
                })

                cm = confusion_matrix(y_test, y_pred)
                st.plotly_chart(px.imshow(cm, text_auto=True, title="Confusion Matrix"))

            else:
                mae = mean_absolute_error(y_test, y_pred)
                rmse = mean_squared_error(y_test, y_pred, squared=False)
                r2 = r2_score(y_test, y_pred)

                results.append({
                    "Model": name,
                    "MAE": mae,
                    "RMSE": rmse,
                    "R2": r2
                })

        results_df = pd.DataFrame(results)
        st.subheader("Comparatie modele")
        st.dataframe(results_df)

        metric = st.selectbox("Metrica pentru best model", results_df.columns[1:])
        best_model = results_df.sort_values(metric, ascending=False).iloc[0]["Model"]

        st.success(f"🏆 Best model: {best_model}")
