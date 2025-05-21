import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Les inn data
df = pd.read_csv('europe-kopi.csv')

#Velg bare de numeriske variablene
features = ['Area', 'GDP', 'Inflation', 'Life.expect', 'Military', 'Pop.growth', 'Unemployment']
X = df[features]

# Standardiser variablene
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Kjør PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

#PCA-resultater
explained_var = pca.explained_variance_ratio_
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(features))], index=features)

# Visualiser PC1 vs PC2
plt.figure(figsize=(10,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1])
for i, country in enumerate(df['Country']):
    plt.text(X_pca[i, 0], X_pca[i, 1], country, fontsize=8)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA – Europe')
plt.grid(True)
plt.show()

# Tolking av PC1
print("Forklart varians (PC1):", explained_var[0])
print("Loadings for PC1:")
print(loadings['PC1'].sort_values(ascending=False))
