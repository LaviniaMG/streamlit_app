import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("EDA – Titanic Dataset")

# =========================
# CERINTA 1 – Incarcare date
# =========================
file = st.file_uploader("Incarca fisier CSV (Titanic)", type="csv")

if file is not None:
    df = pd.read_csv(file)

    st.success("Fisier incarcat cu succes")

    st.subheader("Primele 10 randuri din dataset")
    st.dataframe(df.head(10))

    # Identificare coloane numerice si categorice
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    df_filtrat = df.copy()

    st.subheader("Filtrare date")

    # Slidere pentru coloane numerice
    for col in numeric_cols:
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        interval = st.slider(f"Filtrare {col}", min_val, max_val, (min_val, max_val))
        df_filtrat = df_filtrat[(df_filtrat[col] >= interval[0]) & (df_filtrat[col] <= interval[1])]

    # Multiselect pentru coloane categorice
    for col in categorical_cols:
        valori = st.multiselect(f"Selecteaza valori pentru {col}", df[col].dropna().unique())
        if valori:
            df_filtrat = df_filtrat[df_filtrat[col].isin(valori)]

    st.write("Numar randuri initial:", df.shape[0])
    st.write("Numar randuri dupa filtrare:", df_filtrat.shape[0])

    st.subheader("Dataset filtrat")
    st.dataframe(df_filtrat)

    # =========================
    # CERINTA 2 – Informatii generale
    # =========================
    st.header("Structura datasetului")

    st.write("Numar randuri:", df.shape[0])
    st.write("Numar coloane:", df.shape[1])

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
    fig, ax = plt.subplots()
    sns.barplot(x=missing_df["Coloana"], y=missing_df["Procent (%)"], ax=ax)
    plt.xticks(rotation=90)
    st.pyplot(fig)

    st.subheader("Statistici descriptive (coloane numerice)")
    st.dataframe(df[numeric_cols].describe())

    # =========================
    # CERINTA 3 – Analiza coloana numerica
    # =========================
    st.header("Analiza coloana numerica")

    num_col = st.selectbox("Selecteaza coloana numerica", numeric_cols)
    bins = st.slider("Numar de bins", 10, 100, 30)

    fig, ax = plt.subplots()
    ax.hist(df[num_col].dropna(), bins=bins)
    ax.set_title(f"Histograma pentru {num_col}")
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.boxplot(y=df[num_col], ax=ax)
    ax.set_title(f"Boxplot pentru {num_col}")
    st.pyplot(fig)

    st.write("Media:", df[num_col].mean())
    st.write("Mediana:", df[num_col].median())
    st.write("Deviația standard:", df[num_col].std())

    # =========================
    # CERINTA 4 – Analiza coloana categorica
    # =========================
    st.header("Analiza coloana categorica")

    cat_col = st.selectbox("Selecteaza coloana categorica", categorical_cols)

    freq = df[cat_col].value_counts().reset_index()
    freq.columns = [cat_col, "Frecventa"]
    freq["Procent (%)"] = freq["Frecventa"] / len(df) * 100

    st.dataframe(freq)

    fig, ax = plt.subplots()
    sns.barplot(x=freq[cat_col], y=freq["Frecventa"], ax=ax)
    plt.xticks(rotation=45)
    ax.set_title(f"Distribuția pentru {cat_col}")
    st.pyplot(fig)

    # =========================
    # CERINTA 5 – Corelatii si outlieri
    # =========================
    st.header("Corelatii si outlieri")

    corr = df[numeric_cols].corr()
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    x_col = st.selectbox("Variabila X", numeric_cols)
    y_col = st.selectbox("Variabila Y", numeric_cols, index=1)

    fig, ax = plt.subplots()
    sns.scatterplot(x=df[x_col], y=df[y_col], ax=ax)
    ax.set_title(f"Scatter plot: {x_col} vs {y_col}")
    st.pyplot(fig)

    pearson = df[[x_col, y_col]].corr().iloc[0, 1]
    st.write("Coeficient Pearson:", pearson)

    st.subheader("Detectie outlieri (metoda IQR)")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]

        st.write(f"{col}: {len(outliers)} outlieri ({len(outliers)/len(df)*100:.2f}%)")

        fig, ax = plt.subplots()
        sns.boxplot(y=df[col], ax=ax)
        ax.set_title(f"Outlieri pentru {col}")
        st.pyplot(fig)

else:
    st.info("Incarca fisierul Titanic (train.csv) pentru a incepe analiza")
