
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from mlxtend.frequent_patterns import apriori, association_rules
import plotly.express as px

st.title("📦 Food Packaging Analytics Platform")

# Upload dataset
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview", df.head())

    # Basic preprocessing
    df = df.fillna("Unknown")
    le = LabelEncoder()

    # Encode categorical
    df_encoded = df.copy()
    for col in df.columns:
        if df[col].dtype == 'object':
            df_encoded[col] = le.fit_transform(df[col].astype(str))

    # ---------------- Classification ----------------
    st.header("🤖 Classification: Predict Interest")

    X = df_encoded.drop("Interest", axis=1)
    y = df_encoded["Interest"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    # Feature importance
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
    fig = px.bar(feat_df, x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig)

    # ---------------- Clustering ----------------
    st.header("📊 Customer Segmentation (K-Means)")
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(X)
    df["Cluster"] = clusters

    fig2 = px.histogram(df, x="Cluster", title="Cluster Distribution")
    st.plotly_chart(fig2)

    # ---------------- Regression ----------------
    st.header("📈 Regression: Predict Monthly Loss")
    if "Monthly_Loss" in df_encoded.columns:
        y_reg = df_encoded["Monthly_Loss"]
        X_reg = df_encoded.drop("Monthly_Loss", axis=1)

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2)

        reg = LinearRegression()
        reg.fit(X_train_r, y_train_r)

        st.write("Regression Model Built")

    # ---------------- Association ----------------
    st.header("🔗 Association Rules")

    # Convert multi-select columns
    multi_cols = ["Packaging_Types", "Challenges", "Preferences", "Features_Needed"]

    df_assoc = df[multi_cols].copy()
    df_assoc = df_assoc.apply(lambda x: x.str.get_dummies(sep=", ")).groupby(level=0, axis=1).max()

    frequent = apriori(df_assoc, min_support=0.1, use_colnames=True)
    rules = association_rules(frequent, metric="confidence", min_threshold=0.5)

    st.write(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    # ---------------- New Customer Prediction ----------------
    st.header("📥 Upload New Customer Data")

    new_file = st.file_uploader("Upload new customer CSV", type=["csv"], key="new")

    if new_file:
        new_df = pd.read_csv(new_file)
        new_df = new_df.fillna("Unknown")

        for col in new_df.columns:
            if new_df[col].dtype == 'object':
                new_df[col] = le.fit_transform(new_df[col].astype(str))

        preds = model.predict(new_df)
        st.write("Predictions:", preds)
