# RTide
A Python implementation of the ML [Response Framework: Tidal analysis and prediction through physics-informed ML](https://www.researchsquare.com/article/rs-3289185/v1). The method is an extension of the response method created by Munk and Cartwright in: 'Tidal Spectroscopy and Prediction'. RTide can be used for conventional tidal analysis and prediction, and is well suited to problems involving non-stationary contamination such as tidal rivers, and meteorological forcing. 

# Installation
```
pip install rtide
```
# Usage
## Standard Analysis
Suppose we have a pandas dataframe containing sea-level measurements with a date-time index containing the time of each measurement. A complete RTide analysis can be run in just three lines of code: 
```
df = pd.DataFrame({'observations': observations}, index = times)

model = RTide(df, lat, lon)
model.Prepare_Inputs()
model.Train()
```
A model has now been trained, and the response approximated directly from the provided data. We can now use the learned model to generate predictions whenever we want. All we need is to provide a pandas time-index with the desired times:
```
model.Predict(test_df)
```
## Multivariate Analysis
Now, suppose that we know that our reference site is 'contaminated' by some form of non-stationary forcing (e.g. river outflow). RTide can learn the coupled response to these exogenous inputs directly by simply passing them as an additional column in the input dataframe:
```
df = pd.DataFrame({'observations': observations, 'exogenous_data':river_outflow}, index = times)
model = RTide(df, lat, lon)
...
```
The syntax for the remaining analysis is identical to before. However, several customizations are possible including the selection of custom lags for the multivariate inputs. This is useful in situtations where the dynamics of the interactions with exogenous inputs act on different timescales than the tidal response. 

## Interpretability
The trained RTide model can provide interesting insights into the dynamics of both tidal and non-tidal processes. Standard visualizations for the trained models are provided using the built in 'Visualize_Predictions()' and 'Visualize_Residuals' functions. Physical insights can be obtained using shap analysis as shown in [Response Framework: Tidal analysis and prediction through physics-informed ML](https://www.researchsquare.com/article/rs-3289185/v1). Complete SHAP analysis of the learned model can be performed by calling:
```
model.Shap_Analysis()
```

# Contributions
If you have used RTide for a usecase which is not already covered in the examples section then please reach out by creating an issue. 

# Citing RTide
Bibtex:
```
@article{monahan2023response,
  title={Response Framework: Tidal analysis and prediction through physics-informed ML},
  author={Monahan, Thomas and Tang, Tianning and Roberts, Stephen and Adcock, Thomas},
  year={2023}
}
```

# Useful Response Method References
[1] Munk, Walter Heinrich, and David Edgar Cartwright. "Tidal spectroscopy and prediction." Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences 259.1105 (1966): 533-581.

[2] Cartwright, David Edgar. "A unified analysis of tides and surges round north and east Britain." Philosophical Transactions of the Royal Society of London. Series A, Mathematical and Physical Sciences 263.1134 (1968): 1-55.

[3] Zetler, Bernard, David Cartwright, and Saul Berkman. "Some comparisons of response and harmonic tide predictions." The International Hydrographic Review (1979).

[4] Zetler, Bernard D., and Walter H. Munk. 1975. "The optimum wiggliness of tidal admittances." Journal of Marine Research 33, (S). 

[5] Zetler, Bernard D. "The Evolution of Modern Tide Analysis and Prediction—Some Personal Memories." The International Hydrographic Review (1987).

[6] Amin, M., and G. W. Lennon. "On anomalous tides in Australian waters." The International Hydrographic Review (1992).
