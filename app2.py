import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

# Set page title and icon
st.set_page_config(page_title="Amazon Market Basket Analysis", page_icon="🛒")

# Load the dataset
@st.cache_data
def load_data():
    file_path = "uber-eats-deliveries.csv"  # Make sure this file is uploaded to GitHub

    # Check if file exists
    if not os.path.exists(file_path):
        st.error("⚠️ File not found! Please upload 'uber-eats-deliveries.csv' to your GitHub repository.")
        return None

    df = pd.read_csv(file_path)

    # Clean and preprocess data
    df['discounted_price'] = df['discounted_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['actual_price'] = df['actual_price'].astype(str).str.replace('₹', '').str.replace(',', '').astype(float)
    df['discount_percentage'] = df['discount_percentage'].astype(str).str.replace('%', '').astype(float)
    df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
    df['rating_count'] = df['rating_count'].astype(str).str.replace(',', '').astype(float)

    return df

df = load_data()

# Stop execution if file is missing
if df is None:
    st.stop()

st.title("🛒 Amazon Market Basket Analysis")

# Show dataset preview
st.write("### Sample Data")
st.dataframe(df.head())

# Clustering: Customer Segmentation
st.write("### Customer Segmentation")

features = df[['discounted_price', 'actual_price', 'rating', 'rating_count']].fillna(0)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
df['customer_segment'] = kmeans.fit_predict(scaled_features)

fig, ax = plt.subplots()
sns.scatterplot(x=df['discounted_price'], y=df['actual_price'], hue=df['customer_segment'], palette='viridis', ax=ax)
st.pyplot(fig)

# Association Rule Mining
st.write("### Market Basket Analysis")

basket = df.groupby(['user_id', 'product_name'])['category'].count().unstack().reset_index().fillna(0)
basket.set_index('user_id', inplace=True)
basket = basket.applymap(lambda x: 1 if x > 0 else 0)

frequent_itemsets = apriori(basket, min_support=0.005, use_colnames=True)

if frequent_itemsets.empty:
    st.warning("No frequent itemsets found with the current min_support. Try lowering it further.")
else:
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
    st.write("### Association Rules")
    st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])
