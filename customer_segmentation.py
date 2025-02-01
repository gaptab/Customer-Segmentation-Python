import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Step 1: Generate Dummy Data
np.random.seed(42)
num_customers = 500

data = {
    "CustomerID": np.arange(1, num_customers + 1),
    "Age": np.random.randint(18, 75, num_customers),
    "Annual_Income": np.random.randint(20000, 150000, num_customers),
    "Spending_Score": np.random.randint(1, 100, num_customers),
    "Tenure": np.random.randint(1, 20, num_customers),  # Years with the company
    "Num_Products": np.random.randint(1, 6, num_customers),  # Number of products owned
    "Customer_Interactions": np.random.randint(5, 100, num_customers),  # Number of interactions in a year
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
features = ["Age", "Annual_Income", "Spending_Score", "Tenure", "Num_Products", "Customer_Interactions"]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[features])

# Step 3: Apply K-Means Clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
df["Segment"] = kmeans.fit_predict(df_scaled)

# Step 4: Visualize Clusters using PCA
pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

# Convert PCA results into a DataFrame
pca_df = pd.DataFrame(df_pca, columns=["PCA1", "PCA2"])
pca_df["Segment"] = df["Segment"]  # Ensure segment labels are correctly mapped

# Plot the clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PCA1", y="PCA2", hue="Segment", palette="viridis", s=100)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend(title="Segment")
plt.show()

# Step 5: Save Data to CSV
df.to_csv("customer_segmentation.csv", index=False)

print("Customer segmentation completed. Data saved to 'customer_segmentation.csv'.")
