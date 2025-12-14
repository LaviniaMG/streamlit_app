import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="EDA Titanic", layout="wide")

# =========================
# Fisier default
# =========================
DEFAULT_FILE = "titanic.csv"

# =========================
# Sidebar cu meniu
# =========================
st.sidebar.title("Navigare")
menu = ["Upload & Filtrare", "Structura Dataset", "Analiza Numerica",
        "Analiza Categorica", "Corelatii si Outlieri"]
page = st.sidebar.radio("Selecteaza pagina:", menu)

# =========================
# Pagina 1 – Upload & Filtrare
# =========================
if page == "Upload & Filtrare":
    st.title("EDA – Titanic Dataset")

    file = st.file_uploader("Incarca fisier CSV sau Excel (Titanic)", type=["csv", "xlsx"])

    # initializare session_state pentru df
    if "df" not in st.session_state:
        st.session_state.df = None

    # incarcare fisier
    if file is not None:
        if file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(file)
        else:
            st.session_state.df = pd.read_excel(file)
        st.success("Fisier incarcat cu succes!")
    else:
        if os.path.exists(DEFAULT_FILE) and st.session_state.df is None:
            st.session_state.df = pd.read_csv(DEFAULT_FILE)
            st.info("Se foloseste fisierul default")

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Primele 10 randuri")
        st.dataframe(df.head(10))

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        df_filtrat = df.copy()

        with st.expander("Filtrare coloane numerice"):
            for col in numeric_cols:
                min_val, max_val = float(df[col].min()), float(df[col].max())
                interval = st.slider(f"{col}", min_val, max_val, (min_val, max_val))
                df_filtrat = df_filtrat[(df_filtrat[col] >= interval[0]) & (df_filtrat[col] <= interval[1])]

        with st.expander("Filtrare coloane categorice"):
            for col in categorical_cols:
                valori = st.multiselect(f"{col}", df[col].dropna().unique())
                if valori:
                    df_filtrat = df_filtrat[df_filtrat[col].isin(valori)]

        st.subheader("Numar randuri")
        st.write("Initial:", df.shape[0])
        st.write("Dupa filtrare:", df_filtrat.shape[0])

        st.subheader("Dataset filtrat")
        st.dataframe(df_filtrat)

# =========================
# Pagini urmatoare
# =========================
if page != "Upload & Filtrare":
    df = st.session_state.get("df", None)
    if df is None:
        st.info("Incarca fisierul pe pagina 'Upload & Filtrare' pentru a vizualiza aceasta sectiune.")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        categorical_cols = df.select_dtypes(include="object").columns.tolist()

        # =========================
        # Structura Dataset
        # =========================
        if page == "Structura Dataset":
            st.header("Structura Dataset")

            col1, col2 = st.columns(2)
            col1.metric("Numar randuri", df.shape[0])
            col2.metric("Numar coloane", df.shape[1])

            st.subheader("Tipuri de date")
            st.dataframe(pd.DataFrame({"Coloana": df.columns, "Tip": df.dtypes}))

            st.subheader("Valori lipsa")
            missing = df.isnull().sum()
            missing_pct = (missing / len(df)) * 100
            missing_df = pd.DataFrame({
                "Coloana": df.columns,
                "Valori lipsa": missing.values,
                "Procent (%)": missing_pct.values
            })
            st.dataframe(missing_df)

            st.subheader("Vizualizare valori lipsa")
            fig = px.bar(missing_df, x="Coloana", y="Procent (%)", text="Procent (%)",
                         labels={"Procent (%)": "% valori lipsa"}, title="Procent valori lipsa per coloana")
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("Statistici descriptive")
            st.dataframe(df[numeric_cols].describe().T)

        # =========================
        # Analiza Numerica
        # =========================
        elif page == "Analiza Numerica":
            st.header("Analiza Coloane Numerice")
            num_col = st.selectbox("Selecteaza coloana numerica", numeric_cols)
            bins = st.slider("Numar de bins", 10, 100, 30)

            col1, col2 = st.columns(2)
            fig_hist = px.histogram(df, x=num_col, nbins=bins, marginal="box",
                                    title=f"Distributia pentru {num_col}")
            col1.plotly_chart(fig_hist, use_container_width=True)

            col2.metric("Media", f"{df[num_col].mean():.2f}")
            col2.metric("Mediana", f"{df[num_col].median():.2f}")
            col2.metric("Deviația standard", f"{df[num_col].std():.2f}")

        # =========================
        # Analiza Categorica
        # =========================
        elif page == "Analiza Categorica":
            st.header("Analiza Coloane Categorice")
            cat_col = st.selectbox("Selecteaza coloana categorica", categorical_cols)
            freq = df[cat_col].value_counts().reset_index()
            freq.columns = [cat_col, "Frecventa"]
            freq["Procent (%)"] = freq["Frecventa"] / len(df) * 100

            col1, col2 = st.columns(2)
            col1.dataframe(freq)
            fig_bar = px.bar(freq, x=cat_col, y="Frecventa", text="Frecventa",
                             title=f"Distribuția pentru {cat_col}")
            fig_bar.update_layout(xaxis_tickangle=-45)
            col2.plotly_chart(fig_bar, use_container_width=True)

        # =========================
        # Corelatii si Outlieri
        # =========================
        elif page == "Corelatii si Outlieri":
            st.header("Corelatii si Outlieri")

            st.subheader("Matrice de corelatie")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                                 title="Heatmap corelatii")
            st.plotly_chart(fig_corr, use_container_width=True)

            st.subheader("Scatter Plot")
            x_col = st.selectbox("Variabila X", numeric_cols)
            y_col = st.selectbox("Variabila Y", numeric_cols, index=1)
            fig_scatter = px.scatter(df, x=x_col, y=y_col, trendline="ols",
                                     title=f"{x_col} vs {y_col}")
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Detectie outlieri (IQR)")
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers = df[(df[col] < lower) | (df[col] > upper)]
                st.write(f"{col}: {len(outliers)} outlieri ({len(outliers)/len(df)*100:.2f}%)")
                fig_box = px.box(df, y=col, points="outliers", title=f"Outlieri pentru {col}")
                st.plotly_chart(fig_box, use_container_width=True)
