import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from pyclustering.cluster.kmedoids import kmedoids
from pyclustering.utils import distance_metric, type_metric
import numpy as np

# -------------------------------
# تحميل وتجهيز البيانات
# -------------------------------
df = pd.read_excel("StoresData.xlsx")



# Encoding بسيط
df_encoded = pd.DataFrame()
df_encoded['HomeDel_Yes'] = df['HomeDel'].apply(lambda x: 1 if x == 'Del:Yes' else 0)
df_encoded['Sunday_Open'] = df['Sundays'].apply(lambda x: 1 if x == 'Open' else 0)
df_encoded['State_NSW'] = df['State'].apply(lambda x: 1 if x == 'NSW' else 0)
df_encoded['State_Vic'] = df['State'].apply(lambda x: 1 if x == 'Vic' else 0)
df_encoded['State_Qld'] = df['State'].apply(lambda x: 1 if x == 'Qld' else 0)
df_encoded['Location_Mall'] = df['Location'].apply(lambda x: 1 if x == 'Mall' else 0)
df_encoded['Location_Country'] = df['Location'].apply(lambda x: 1 if x == 'Country' else 0)
df_encoded['Location_Strip'] = df['Location'].apply(lambda x: 1 if x == 'Strip' else 0)
df_encoded['Manager_Male'] = df['Mng-Sex'].apply(lambda x: 1 if x == 'M' else 0)
df_encoded['Manager_Female'] = df['Mng-Sex'].apply(lambda x: 1 if x == 'F' else 0)

# Association Rules
frequent_itemsets = apriori(df_encoded, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.0)

# Clustering - K-Medoids
numerical_cols = [
    'Sales $m', 'Wages $m', 'No. Staff', 'Loc\'n (Num)', 'State (Num)',
    'Age (Yrs)', 'GrossProfit', 'Adv.$\'000', 'Competitors', 'HrsTrading',
    'Sundays (Num)', 'Mng-Sex (Num)', 'Mng-Age', 'Mng-Exp', 'Mng-Train',
    'Union%', 'Car Spaces', 'HomeDel (Num)', 'Basket:2013', 'Basket:2014'
]
df_cluster = df[numerical_cols]
data_full = df_cluster.values

initial_medoids = [0, 1, 2]
metric = distance_metric(type_metric.EUCLIDEAN)
kmedoids_instance = kmedoids(data_full, initial_medoids, metric=metric)
kmedoids_instance.process()
clusters = kmedoids_instance.get_clusters()

labels = np.zeros(len(df))
for cluster_idx, cluster in enumerate(clusters):
    for index in cluster:
        labels[index] = cluster_idx
df['Cluster'] = labels

# -------------------------------
# Functions
# -------------------------------
def recommend(features_input):
    recommendations = []
    if isinstance(features_input, str):
        features_input = [features_input]
    for idx, rule in rules.iterrows():
        if any(feature in rule['antecedents'] for feature in features_input):
            for consequent in rule['consequents']:
                if consequent not in features_input:
                    recommendations.append((list(rule['antecedents']), consequent, rule['confidence'], rule['lift']))
    if not recommendations:
        return "No recommendations found."
    recommendations = sorted(recommendations, key=lambda x: x[2], reverse=True)
    return pd.DataFrame(recommendations, columns=['Input Feature(s)', 'Recommended Feature', 'Confidence', 'Lift'])

# -------------------------------
# Streamlit Interface
# -------------------------------
st.set_page_config(page_title="Store Analysis Dashboard", page_icon="🏪", layout="wide")

st.title("🏪 Store Analysis and Recommendation Dashboard")

# Summary
st.subheader("Summary Info")
col1, col2 = st.columns(2)
col1.metric("Total Stores", len(df))
col2.metric("Number of Clusters", len(set(labels)))

st.divider()

# Charts Section
st.subheader("Visualizations")

# Pie Chart
st.write("### Stores Distribution by Cluster")
cluster_counts = df['Cluster'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(cluster_counts, labels=cluster_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
ax1.axis('equal')
st.pyplot(fig1)

# Scatter Plot
st.write("### Sales vs Wages (Colored by Cluster)")
fig2, ax2 = plt.subplots()
sns.scatterplot(x=df['Sales $m'], y=df['Wages $m'], hue=df['Cluster'].astype(int), palette="Set2", ax=ax2)
plt.xlabel("Sales ($m)")
plt.ylabel("Wages ($m)")
plt.title("Scatter Plot of Sales vs Wages")
plt.grid(True)
st.pyplot(fig2)

st.divider()

# Recommendation Section
st.subheader("Get Store Recommendations 🔍")

features = ['HomeDel_Yes', 'Sunday_Open', 'State_NSW', 'State_Vic', 'State_Qld', 'Location_Mall', 'Location_Country', 'Location_Strip', 'Manager_Male', 'Manager_Female']
user_choice = st.multiselect("Select existing store features:", features)

if user_choice:
    st.write("### Recommended Features:")
    result = recommend(user_choice)
    st.dataframe(result)
else:
    st.info("Please select at least one feature to get recommendations.")

