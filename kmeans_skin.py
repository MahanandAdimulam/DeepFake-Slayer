import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load CSV file
csv_file = '/shared/rc/defake/Deepfake-Slayer/output/test/test_list_color_values.csv'
data = pd.read_csv(csv_file)

# Select features
X = data[['Avg_Lab_L']]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Map cluster labels to skin color categories
cluster_to_skin_color = {
    0: 'Light',
    1: 'Medium',
    2: 'Dark'
}

# Add skin color column to the DataFrame
data['Skin_Color_L'] = [cluster_to_skin_color[label] for label in clusters]

# Visualize clusters
plt.figure(figsize=(10, 6))
for cluster, color, label in zip(range(3), ['blue', 'green', 'orange'], ['Light', 'Medium', 'Dark']):
    cluster_data = X_scaled[clusters == cluster]
    plt.hist(cluster_data, bins=20, alpha=0.6, color=color, label=label)

plt.title('Distribution of Clusters for Avg_Lab_L')
plt.xlabel('Standardized Avg_Lab_L')
plt.ylabel('Frequency')
plt.legend()
plt.grid()
plt.savefig('/shared/rc/defake/Deepfake-Slayer/output/test/KMeans_Clusters.png')

# Save the clustered data
csv_file = '/shared/rc/defake/Deepfake-Slayer/output/test/test_list_color_values_updated.csv'
data.to_csv(csv_file, index=False)
print("Clustering complete. Clustered data saved to 'clustered_data.csv'")

df = pd.read_csv(csv_file)

# Step 2: Define the mapping dictionary
skin_color_mapping = {'Light': 0, 'Medium': 1, 'Dark': 2, 'Unknown': -1}

# Step 3: Map the 'skin_color' column to create 'skin_color_encoded'
df['Skin_Color_Encoded'] = df['Skin_Color_L'].map(skin_color_mapping)

# Step 4: Save the updated DataFrame back to a CSV file
df.to_csv(csv_file, index=False)

print(f"Updated CSV saved at: {csv_file}")