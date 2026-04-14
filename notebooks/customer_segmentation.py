# ─────────────────────────────────────────
# Customer Segmentation | Olist E-Commerce
# ─────────────────────────────────────────

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
import os

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════
# PHASE 1 — LOAD DATA
# ═══════════════════════════════════════════

BASE_PATH = r"D:\Document\AinGenX DATA ANALYST\Python\Projects\customer-segmentation-project\data"

orders      = pd.read_csv(os.path.join(BASE_PATH, "olist_orders_dataset.csv"))
payments    = pd.read_csv(os.path.join(BASE_PATH, "olist_order_payments_dataset.csv"))
customers   = pd.read_csv(os.path.join(BASE_PATH, "olist_customers_dataset.csv"))
order_items = pd.read_csv(os.path.join(BASE_PATH, "olist_order_items_dataset.csv"))

print("Data loaded successfully\n")

for name, df in {"orders": orders, "payments": payments,
                 "customers": customers, "order_items": order_items}.items():
    print(f"  {name:15} → {df.shape[0]:,} rows  |  {df.shape[1]} columns")

    # ═══════════════════════════════════════════
# PHASE 2 — MERGE & CLEAN DATA
# ═══════════════════════════════════════════

# Keep only delivered orders
orders_delivered = orders[orders['order_status'] == 'delivered'].copy()

# Merge orders + customers
df = orders_delivered.merge(customers, on='customer_id', how='left')

# Merge + payments
df = df.merge(payments, on='order_id', how='left')

# Merge + order items
df = df.merge(order_items[['order_id', 'price']], on='order_id', how='left')

# Convert date column to datetime
df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])

# Drop rows with missing payment values
df.dropna(subset=['payment_value'], inplace=True)

print(f"Data merged & cleaned → {df.shape[0]:,} rows | {df.shape[1]} columns\n")

# ═══════════════════════════════════════════
# PHASE 3 — RFM FEATURE ENGINEERING
# ═══════════════════════════════════════════

snapshot_date = df['order_purchase_timestamp'].max() + pd.Timedelta(days=1)

rfm = df.groupby('customer_unique_id').agg(
    Recency   = ('order_purchase_timestamp', lambda x: (snapshot_date - x.max()).days),
    Frequency = ('order_id', 'nunique'),
    Monetary  = ('payment_value', 'sum')
).reset_index()

print("RFM table created\n")
print(rfm.describe())

# ═══════════════════════════════════════════
# PHASE 4 — K-MEANS CLUSTERING
# ═══════════════════════════════════════════

# Scale the RFM features
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

# Elbow method to find best K
inertia = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(rfm_scaled)
    inertia.append(km.inertia_)

# Plot elbow curve
plt.figure(figsize=(8, 4))
plt.plot(k_range, inertia, marker='o', color='steelblue')
plt.title('Elbow Method — Optimal K')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.tight_layout()
os.makedirs("visuals", exist_ok=True)
plt.savefig("visuals/elbow_curve.png")
plt.close()
print("Elbow curve saved\n")

# Train final model with K=4
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
rfm['Cluster'] = kmeans.fit_predict(rfm_scaled)

print(f"Clusters assigned\n")
print(rfm['Cluster'].value_counts())

# ═══════════════════════════════════════════
# PHASE 5 — SEGMENT LABELING
# ═══════════════════════════════════════════

# Analyze cluster averages
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
print("\n Cluster Summary:\n")
print(cluster_summary)

# Label each cluster based on RFM profile
# After seeing cluster_summary, we map labels accordingly
segment_map = {
    rfm.groupby('Cluster')['Monetary'].mean().idxmax()  : 'Champions',
    rfm.groupby('Cluster')['Recency'].mean().idxmin()   : 'New Customers',
    rfm.groupby('Cluster')['Recency'].mean().idxmax()   : 'Lost',
    rfm.groupby('Cluster')['Frequency'].mean().idxmax() : 'Loyal Customers'
}

rfm['Segment'] = rfm['Cluster'].map(segment_map).fillna('At-Risk')

print("\n Segments assigned\n")
print(rfm['Segment'].value_counts())

# ═══════════════════════════════════════════
# PHASE 6 — VISUALIZATIONS
# ═══════════════════════════════════════════

# 1. Customer count per segment
plt.figure(figsize=(8, 5))
order = rfm['Segment'].value_counts().index
sns.countplot(data=rfm, x='Segment', order=order, palette='Set2')
plt.title('Customer Count per Segment')
plt.xlabel('Segment')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("visuals/segment_count.png")
plt.close()
print("Segment count chart saved")

# 2. Recency vs Monetary scatter
plt.figure(figsize=(8, 5))
sns.scatterplot(data=rfm, x='Recency', y='Monetary',
                hue='Segment', palette='Set1', alpha=0.6)
plt.title('Recency vs Monetary by Segment')
plt.tight_layout()
plt.savefig("visuals/recency_vs_monetary.png")
plt.close()
print("Scatter plot saved")

# 3. RFM heatmap per segment
plt.figure(figsize=(8, 4))
heatmap_data = rfm.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(2)
sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlGnBu')
plt.title('Average RFM Values per Segment')
plt.tight_layout()
plt.savefig("visuals/rfm_heatmap.png")
plt.close()
print("Heatmap saved\n")

# ═══════════════════════════════════════════
# PHASE 7 — EXPORT OUTPUT
# ═══════════════════════════════════════════

os.makedirs("data", exist_ok=True)
rfm.to_csv("data/customer_segments.csv", index=False)

print("customer_segments.csv exported\n")
print("═" * 40)
print("Project complete!")
print(f"   Total customers segmented : {len(rfm):,}")
print(f"   Segments created          : {rfm['Segment'].nunique()}")
print("═" * 40)