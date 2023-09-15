import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def cosd(x):
  return np.cos(np.deg2rad(x))
def sind(x):
  return np.sin(np.deg2rad(x))

def mfft(x,DeltaTau):
  return np.exp(1j*-2*np.pi*x*DeltaTau)

def pad_dataframe(df, N):
    # Determine the frequency of the existing index values
    freq = pd.infer_freq(df.index)

    if freq is None:
        freq_counts = df.index.to_series().diff().value_counts()
        most_common_freq = freq_counts.idxmax()
        freq = most_common_freq

    # Create a new DataFrame with the desired frequency and UTC timezone
    new_index = pd.date_range(start=df.index[0] - pd.DateOffset(hours=N), end=df.index[-1] + pd.DateOffset(hours=N), freq=freq, tz='UTC')
    padded_df = pd.DataFrame(index=new_index)

    # Merge the existing DataFrame with the padded DataFrame based on the index
    padded_df = padded_df.merge(df, how='left', left_index=True, right_index=True)

    # Fill missing values in the padded DataFrame with zeros
    #padded_df.fillna(0, inplace=True)

    return padded_df

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def dim_reduc(values, plot = False):
  X = np.array(values).T
  scaler = StandardScaler()
  X_standardized = scaler.fit_transform(X)

  # Step 3: Calculate the Covariance Matrix
  covariance_matrix = np.cov(X_standardized, rowvar=False)

  # Step 4: Eigenvalue Decomposition
  eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

  pca = PCA().fit(X_standardized)
  explained_variance = pca.explained_variance_ratio_
  cumulative_explained_variance = np.cumsum(explained_variance)
  if plot:
    plt.title(f'Total Variance Explained by {k} components: {cumsum}')
    plt.plot(range(1, len(cumulative_explained_variance) + 1), cumulative_explained_variance)
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()
  ind = 0
  cumsum = cumulative_explained_variance[0]
  while cumsum < .985:
    ind +=1
    cumsum = cumulative_explained_variance[ind]

  k = ind+1  # You can adjust this as needed
  top_eigenvalues_indices = np.argsort(eigenvalues)[::-1][:k]
  selected_eigenvectors = eigenvectors[:, top_eigenvalues_indices]

  # Step 6: Projection
  return np.dot(X_standardized, selected_eigenvectors).T
