## SHAP

import numpy as np
from scipy.stats import t
import shap
def SHAPS(df, global_tide, lats):
  '''
  Note this is the integration for the CNN, implementation for NN is more straightforward.
  '''
  rad_vals = []

  values = []

  dfs_separate = prepare_dfs(df,global_tide, uniform = False, nonuni = 'utide')[24*15:-24*7]
  modela, X_train, scaler_Y_U  = run_conv_shap(dfs_separate, train_days = 30, train_epochs = 10,train_epochs2 = 25, lr_1 = 0.00005, lr_2 = 0.000001, dropout = 0, timestep = 'hour', lati = lats, dim_reduc = 1, nonlin = False)
  def f(x):
    shape1 = np.shape(x)[0]
    shape2 = np.shape(x)[1]
    return scaler_Y_U.inverse_transform(modela.predict(x.reshape(shape1,shape2,1)))

  shape1 = np.shape(X_train)[0]
  shape2 = np.shape(X_train)[1]
  #explainer = shap.KernelExplainer(f, X_train.reshape(shape1,shape2)[:100,:])
  explainer = shap.KernelExplainer(f, shap.sample(X_train.reshape(shape1,shape2),30))
  randomlist = random.sample(range(len(X_train)), 30)
  shap_values = explainer.shap_values(X_train.reshape(shape1,shape2)[randomlist], nsamples=250)
  mean_arr = np.mean(np.abs(shap_values), axis=1) ## Corrected so that these are absolute values
  return mean_arr




def get_significance(shap_samples, verbose = False):
  #shap_samples = np.array([run1, run2,run3,run4,run5,run6,run7,run8,run9,run10])

  # Assuming shap_samples is a list of length 10, where each element is a list of 9 feature importance values
  # For example:
  # shap_samples = [[sample1_feature1, sample1_feature2, ..., sample1_feature9],
  #                 [sample2_feature1, sample2_feature2, ..., sample2_feature9],
  #                 ...,
  #                 [sample10_feature1, sample10_feature2, ..., sample10_feature9]]

  # Compute mean values and standard deviations for each feature
  mean_values = np.mean(shap_samples, axis=0)
  std_values = np.std(shap_samples, axis=0, ddof=1)  # Use ddof=1 for unbiased estimator of standard deviation

  # Compute the number of samples
  num_samples = len(shap_samples)

  # Compute the standard error of the mean for each feature
  standard_error = std_values / np.sqrt(num_samples)

  # Define the t-distribution critical value for a 99% confidence level (assuming 10 samples)
  t_critical_value = t.ppf(0.995, df=num_samples - 1)

  # Compute the 99% confidence interval for each feature
  lower_bounds = mean_values - t_critical_value * standard_error
  upper_bounds = mean_values + t_critical_value * standard_error

  # Check if there is any overlap between the confidence intervals
  overlap = np.any((lower_bounds[:, np.newaxis] <= upper_bounds) & (upper_bounds[:, np.newaxis] >= lower_bounds), axis=0)

  significant_features = []
  for feature_idx in range(len(mean_values)):
      significant = True
      for other_feature_idx in range(len(mean_values)):
          if feature_idx != other_feature_idx:
              if lower_bounds[feature_idx] <= upper_bounds[other_feature_idx] and upper_bounds[feature_idx] >= lower_bounds[other_feature_idx]:
                  significant = False
                  break
      if significant:
          significant_features.append(feature_idx)



  # Print the results
  if verbose:
    print("Feature\tMean\t\t99% Confidence Interval")
  for feature_idx, mean_value in enumerate(mean_values):
      significant = "Significant" if feature_idx in significant_features else "Not Significant"
      if verbose:
        print(f"Feature {feature_idx + 1}\t{mean_value:.4f}\t\t({lower_bounds[feature_idx]:.4f}, {upper_bounds[feature_idx]:.4f})\t\t{significant}")
  return mean_values, lower_bounds, upper_bounds, significant_features
