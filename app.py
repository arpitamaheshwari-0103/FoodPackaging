import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

from mlxtend.frequent_patterns import apriori, association_rules

import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Food Packaging Analytics", layout="wide")

st.title("📦 Food Packaging Analytics Platform")

uploaded_file = st.file_uploader("Upload Dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    df = df.fillna("Unknown")

    df_encoded = df.copy()
    le_dict = {}

    for col in df.columns:
        if df[col].dtype == "object":
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    st.header("🤖 Classification: Predict Interest")

    X = df_encoded.drop("Interest", axis=1)
    y = df_encoded["Interest"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    st.write("### Performance Metrics")
    st.write("Accuracy:", acc)
    st.write("Precision:", prec)
    st.write("Recall:", rec)
    st.write("F1 Score:", f1)

    if len(np.unique(y)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'))
        fig.add_shape(type='line', line=dict(dash='dash'), x0=0, x1=1, y0=0, y1=1)

        fig.update_layout(title=f"ROC Curve (AUC = {roc_auc:.2f})",
                          xaxis_title="False Positive Rate",
                          yaxis_title="True Positive Rate")

        st.plotly_chart(fig)

    st.subheader("📊 Feature Importance")
    importance = model.feature_importances_
    feat_df = pd.DataFrame({"Feature": X.columns, "Importance": importance})
    fig_imp = px.bar(feat_df.sort_values(by="Importance", ascending=False),
                     x="Feature", y="Importance", title="Feature Importance")
    st.plotly_chart(fig_imp)

    st.header("📊 Customer Segmentation")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(X)

    df["Cluster"] = clusters

    fig_cluster = px.histogram(df, x="Cluster", title="Customer Segments")
    st.plotly_chart(fig_cluster)

    st.header("📈 Regression: Predict Monthly Loss")

    if "Monthly_Loss" in df_encoded.columns:
        y_reg = df_encoded["Monthly_Loss"]
        X_reg = df_encoded.drop("Monthly_Loss", axis=1)

        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2)

        reg = LinearRegression()
        reg.fit(X_train_r, y_train_r)

        st.success("Regression Model Built Successfully")

    st.header("🔗 Association Rules")

    multi_cols = ["Packaging_Types", "Challenges", "Preferences", "Features_Needed"]

    try:
        df_assoc = df[multi_cols].fillna("").astype(str)

        dummies_list = []

        for col in multi_cols:
            dummies = df_assoc[col].str.get_dummies(sep=", ")
            dummies_list.append(dummies)

        df_final = pd.concat(dummies_list, axis=1)
        df_final = df_final.loc[:, ~df_final.columns.duplicated()]
        df_final = df_final.astype(bool)

        min_support = st.slider("Min Support", 0.01, 0.5, 0.05)
        min_confidence = st.slider("Min Confidence", 0.1, 1.0, 0.4)

        frequent = apriori(df_final, min_support=min_support, use_colnames=True)

        if len(frequent) == 0:
            st.warning("No frequent itemsets found. Try lowering support.")
        else:
            rules = association_rules(frequent, metric="confidence", min_threshold=min_confidence)

            if len(rules) == 0:
                st.warning("No association rules found. Try lowering confidence.")
            else:
                rules = rules.sort_values(by="lift", ascending=False)

                st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

    except Exception as e:
        st.error("Error in Association Rules section")
        st.write(e)

    st.header("📥 Upload New Customer Data")

    new_file = st.file_uploader("Upload new customer CSV", type=["csv"], key="new")

    if new_file:
        new_df = pd.read_csv(new_file)
        new_df = new_df.fillna("Unknown")

        for col in new_df.columns:
            if col in le_dict:
                new_df[col] = le_dict[col].transform(new_df[col].astype(str))

        preds = model.predict(new_df)
        probs = model.predict_proba(new_df)

        result = new_df.copy()
        result["Prediction"] = preds
        result["Probability"] = probs.max(axis=1)

        st.write("### Predictions")
        st.dataframe(result)
