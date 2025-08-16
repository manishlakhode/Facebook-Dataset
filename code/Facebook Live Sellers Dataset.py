# Facebook Live Sellers Dataset Analysis and Clustering

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans

# --- Step 1: Load the Dataset ---
file_path = r'D:\Finlatics\MLResearch\Facebook Dataset\Facebook_Marketplace_data.csv'

try:
    df = pd.read_csv(file_path)
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the file path.")
    exit()

# --- Step 2: Data Preprocessing ---
# Convert status_published to datetime

df['status_published'] = pd.to_datetime(df['status_published'], errors='coerce')

# Handle missing values (drop rows with all NaNs in engagement columns)
df = df.dropna(subset=['num_reactions', 'num_comments', 'num_shares'])

# Extract hour for time analysis
df['hour'] = df['status_published'].dt.hour

# --- Q5: Count of different post types ---
plt.figure(figsize=(8, 4))
sns.countplot(data=df, x='status_type', order=df['status_type'].value_counts().index)
plt.title('Count of Different Post Types')
plt.xlabel('Status Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

print("Count of different types of posts:")
print(df['status_type'].value_counts())
print("="*60) 

# --- Q6: Average engagement per post type ---
avg_metrics_by_type = df.groupby('status_type')[['num_reactions', 'num_comments', 'num_shares']].mean().round(2)
print("Average engagement metrics for each post type:")
print(avg_metrics_by_type)

avg_metrics_by_type.plot(kind='bar', figsize=(10,5))
plt.title('Average Engagement Metrics by Post Type')
plt.ylabel('Average Value')
plt.xticks(rotation=45)
plt.show()

print("="*60)

# --- Q1: Time of Upload vs Number of Reactions ---
average_reactions_by_hour = df.groupby('hour')['num_reactions'].mean().reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x='hour', y='num_reactions', data=average_reactions_by_hour)
plt.title('Average Number of Reactions by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Average Number of Reactions')
plt.grid(axis='y', linestyle='--')
plt.show()

print("="*60)

# --- Q2: Correlation Analysis ---
correlation_columns = ['num_reactions', 'num_comments', 'num_shares']
correlation_matrix = df[correlation_columns].corr()

sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Engagement Metrics')
plt.show()

print("Correlation Matrix:")
print(correlation_matrix)
print("="*60)

# --- Q4: Elbow Method for K-Means ---
clustering_features = df[['status_type', 'num_reactions', 'num_comments', 'num_shares',
                          'num_likes', 'num_loves', 'num_wows', 'num_hahas',
                          'num_sads', 'num_angrys']].copy()

le = LabelEncoder()
clustering_features['status_type_encoded'] = le.fit_transform(clustering_features['status_type'])
clustering_features.drop('status_type', axis=1, inplace=True)

scaler = StandardScaler()
scaled_features = scaler.fit_transform(clustering_features)

inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.xticks(range(1, 11))
plt.grid(True)
plt.show()

print("Based on the elbow plot, choose 3 or 4 clusters.")
print("="*60)

# --- Q3: K-Means Clustering ---
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['cluster'] = kmeans.fit_predict(scaled_features)

print(f"K-Means Clustering Model Trained with {optimal_k} Clusters.")
print("Distribution of posts per cluster:")
print(df['cluster'].value_counts())

# Cluster analysis
cluster_features_df = pd.DataFrame(scaled_features, columns=clustering_features.columns)
cluster_features_df['cluster'] = df['cluster']

print("Mean values of features for each cluster (standardized):")
print(cluster_features_df.groupby('cluster').mean().round(2))

# Cluster-wise post type distribution
df['status_type_encoded'] = clustering_features['status_type_encoded']
df['status_type'] = le.inverse_transform(df['status_type_encoded'])
cluster_type_distribution = df.groupby('cluster')['status_type'].value_counts(normalize=True).unstack().round(2).fillna(0)

print("Cluster Post Type Distribution:")
print(cluster_type_distribution)
