import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression

from mlxtend.frequent_patterns import apriori, association_rules

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Food Packaging Analytics", layout="wide")

st.title("📦 Food Packaging Analytics Platform")

menu = st.sidebar.selectbox(
    "Navigation",
    ["Overview", "Analysis", "Prediction", "Association", "Lead Scoring"]
)

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df = df.fillna("Unknown")

    df_encoded = df.copy()
    le_dict = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    if menu == "Overview":
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head())

    elif menu == "Analysis":
        st.subheader("📊 Feature Importance & Clustering")

        X = df_encoded.drop("Interest", axis=1)
        y = df_encoded["Interest"]

        model = RandomForestClassifier()
        model.fit(X, y)

        importance = model.feature_importances_
        feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})

        fig = px.bar(feat_df.sort_values(by="Importance", ascending=False),
                     x="Feature", y="Importance")
        st.plotly_chart(fig)

        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(X)
        df["Cluster"] = clusters

        fig2 = px.histogram(df, x="Cluster", title="Customer Segments")
        st.plotly_chart(fig2)

    elif menu == "Prediction":
        st.subheader("🤖 Interest Prediction Model")

        X = df_encoded.drop("Interest", axis=1)
        y = df_encoded["Interest"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        model = RandomForestClassifier()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)

        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("Precision:", precision_score(y_test, y_pred, average='weighted'))
        st.write("Recall:", recall_score(y_test, y_pred, average='weighted'))
        st.write("F1:", f1_score(y_test, y_pred, average='weighted'))

        if len(np.unique(y)) == 2:
            fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
            roc_auc = auc(fpr, tpr)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines'))
            fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

            fig.update_layout(title=f"ROC Curve (AUC={roc_auc:.2f})")
            st.plotly_chart(fig)

    elif menu == "Association":
        st.subheader("🔗 Association Rules")

        multi_cols = ["Packaging_Types", "Challenges", "Preferences", "Features_Needed"]

        df_assoc = df[multi_cols].fillna("").astype(str)

        dummies_list = []
        for col in multi_cols:
            dummies = df_assoc[col].str.get_dummies(sep=", ")
            dummies_list.append(dummies)

        df_final = pd.concat(dummies_list, axis=1)
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        df_final = df_final.astype(bool)

        frequent = apriori(df_final, min_support=0.05, use_colnames=True)

        if len(frequent) > 0:
            rules = association_rules(frequent, metric="confidence", min_threshold=0.4)
            rules = rules.sort_values(by="lift", ascending=False)
            st.dataframe(rules[['antecedents','consequents','support','confidence','lift']])
        else:
            st.warning("No rules found")

    elif menu == "Lead Scoring":
        st.subheader("📥 Upload New Customer Data")

        new_file = st.file_uploader("Upload new customer CSV", type=["csv"], key="new")

        if new_file:
            new_df = pd.read_csv(new_file)
            new_df = new_df.fillna("Unknown")

            for col in new_df.columns:
                if col in le_dict:
                    new_df[col] = le_dict[col].transform(new_df[col].astype(str))

            X = df_encoded.drop("Interest", axis=1)
            y = df_encoded["Interest"]

            model = RandomForestClassifier()
            model.fit(X, y)

            preds = model.predict(new_df)
            probs = model.predict_proba(new_df)

            new_df["Prediction"] = preds
            new_df["Probability"] = probs.max(axis=1)

            st.dataframe(new_df)
