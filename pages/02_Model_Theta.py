##################################################
## {Description: Capstone project for DS4A Colombia and Eficacia S.A.}
##################################################
## {License_info: Restricted For educational proporses}
##################################################
## Author: {Team 108 DS4A Colombia}
## Copyright: Copyright {2022}, {Efficacia DS APP}
## Credits: [{credit_list: Team 108 :Andres Felipe Garcia, Camilo Gomez, Carlos Andres Cubillos , Ilan Almaza, Luis Miguel Puerto, Ricardo Leon, Ricardo Rodriguez, ; Data from eficacia S.A.; soport Correlation one DS4A colombia}]
## License: {license}
## Version: {mayor}.{minor}.{rel}
## Maintainer: {Unmaintained}
## Email: {contact_email}
## Status: {dev_status: under development}
##################################################
import streamlit as st
from autots import AutoTS
import matplotlib.pyplot as plt
import pandas as pd



st.title(":chart_with_upwards_trend: Model Theta")


#creating horizontal containers
Formula_us= st.container()
Data__in= st.container()
submenus=st.container()
Model_exp= st.container()
ga_model= st.container()


table_models = """<table>
<thead>
<tr>
<th>Model</th>
<th>Python packages</th>
<th>Parameters and<br> Error values</th>
</tr>
</thead>
<tbody>
<tr>
<td>GLM</td>
<td>only statsmodels</td>
<td>Generalized Linear<br> Model Regression Results Dep. Variable: [&#39;Agotado[False]&#39;, &#39;Agotado[True]&#39;]<br> No. Observations: 65505<br><br> Model: GLM Df Residuals: 65186<br><br> Model Family: Binomial Df Model: 318<br><br> Link Function: Logit Scale: 1.0000<br><br> Method: IRLS Log-Likelihood: -450.41<br><br> Date: Fri, 01 Jul 2022 Deviance: 900.82<br><br> Time: 13:14:19 Pearson chi2: 1.61e+04<br><br> No. Iterations: 30 Pseudo R-squ. (CS): 0.01461<br><br> Covariance Type: nonrobust</td>
</tr>
<tr>
<td>Autots(LastValueNaive&#39;)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> LastValueNaive<br><br> {&#39;fillna&#39;: &#39;fake_date&#39;, &#39;transformations&#39;: {&#39;0&#39;: &#39;SeasonalDifference&#39;, &#39;1&#39;:<br> &#39;QuantileTransformer&#39;, &#39;2&#39;: &#39;StandardScaler&#39;}, &#39;transformation_params&#39;: {&#39;0&#39;:<br> {&#39;lag_1&#39;: 7, &#39;method&#39;: &#39;LastValue&#39;}, &#39;1&#39;: {&#39;output_distribution&#39;: &#39;uniform&#39;,<br> &#39;n_quantiles&#39;: 20}, &#39;2&#39;: {}}}<br><br> {}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308641975308641</td>
</tr>
<tr>
<td>Autots(GLS)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> GLS<br><br> {&#39;fillna&#39;: &#39;ffill&#39;, &#39;transformations&#39;: {&#39;0&#39;: &#39;ClipOutliers&#39;, &#39;1&#39;:<br> &#39;QuantileTransformer&#39;, &#39;2&#39;: &#39;RobustScaler&#39;}, &#39;transformation_params&#39;: {&#39;0&#39;:<br> {&#39;method&#39;: &#39;clip&#39;, &#39;std_threshold&#39;: 3.5, &#39;fillna&#39;: None}, &#39;1&#39;:<br> {&#39;output_distribution&#39;: &#39;uniform&#39;, &#39;n_quantiles&#39;: 44}, &#39;2&#39;: {}}}<br><br> {}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308641975308641</td>
</tr>
<tr>
<td>Autots(GLM)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> Ensemble<br><br> {}<br><br> {&#39;model_name&#39;: &#39;BestN&#39;, &#39;model_count&#39;: 3, &#39;model_metric&#39;: &#39;best_score&#39;,<br> &#39;models&#39;: {&#39;6dfcfc7db1dd3fb2b889d059da9a1c00&#39;: {&#39;Model&#39;: &#39;GLM&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;family&quot;: &quot;NegativeBinomial&quot;,<br> &quot;constant&quot;: false, &quot;regression_type&quot;: null}&#39;,<br> &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;rolling_mean_24&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;IntermittentOccurrence&quot;, &quot;1&quot;:<br> &quot;QuantileTransformer&quot;, &quot;2&quot;:<br> &quot;QuantileTransformer&quot;, &quot;3&quot;: &quot;Detrend&quot;},<br> &quot;transformation_params&quot;: {&quot;0&quot;: {&quot;center&quot;:<br> &quot;midhinge&quot;}, &quot;1&quot;: {&quot;output_distribution&quot;:<br> &quot;normal&quot;, &quot;n_quantiles&quot;: 44}, &quot;2&quot;:<br> {&quot;output_distribution&quot;: &quot;uniform&quot;,<br> &quot;n_quantiles&quot;: 20}, &quot;3&quot;: {&quot;model&quot;:<br> &quot;Linear&quot;, &quot;phi&quot;: 1, &quot;window&quot;: null}}}&#39;},<br> &#39;e6ac7e51e57a98b340af7e91b6d5f9e6&#39;: {&#39;Model&#39;: &#39;GLM&#39;, &#39;ModelParameters&#39;:<br> &#39;{&quot;family&quot;: &quot;Gaussian&quot;, &quot;constant&quot;: true,<br> &quot;regression_type&quot;: null}&#39;, &#39;TransformationParameters&#39;:<br> &#39;{&quot;fillna&quot;: &quot;median&quot;, &quot;transformations&quot;:<br> {&quot;0&quot;: &quot;PowerTransformer&quot;, &quot;1&quot;:<br> &quot;QuantileTransformer&quot;, &quot;2&quot;:<br> &quot;QuantileTransformer&quot;, &quot;3&quot;: &quot;bkfilter&quot;},<br> &quot;transformation_params&quot;: {&quot;0&quot;: {}, &quot;1&quot;:<br> {&quot;output_distribution&quot;: &quot;normal&quot;,<br> &quot;n_quantiles&quot;: 44}, &quot;2&quot;:<br> {&quot;output_distribution&quot;: &quot;uniform&quot;,<br> &quot;n_quantiles&quot;: 20}, &quot;3&quot;: {}}}&#39;}}, &#39;model_weights&#39;: {},<br> &#39;point_method&#39;: &#39;mean&#39;}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308641975308641</td>
</tr>
<tr>
<td>AutoTS(ETS)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> ETS<br><br> {&#39;fillna&#39;: &#39;rolling_mean_24&#39;, &#39;transformations&#39;: {&#39;0&#39;: &#39;ClipOutliers&#39;, &#39;1&#39;:<br> &#39;QuantileTransformer&#39;, &#39;2&#39;: &#39;Detrend&#39;}, &#39;transformation_params&#39;: {&#39;0&#39;:<br> {&#39;method&#39;: &#39;clip&#39;, &#39;std_threshold&#39;: 3, &#39;fillna&#39;: None}, &#39;1&#39;:<br> {&#39;output_distribution&#39;: &#39;uniform&#39;, &#39;n_quantiles&#39;: 44}, &#39;2&#39;: {&#39;model&#39;: &#39;GLS&#39;,<br> &#39;phi&#39;: 1, &#39;window&#39;: None}}}<br><br> {&#39;damped_trend&#39;: False, &#39;trend&#39;: None, &#39;seasonal&#39;: None,<br> &#39;seasonal_periods&#39;: None}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308641975308641</td>
</tr>
<tr>
<td>AutoTS(AverageValueNaive)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> Ensemble<br><br> {}<br><br> {&#39;model_name&#39;: &#39;BestN&#39;, &#39;model_count&#39;: 5, &#39;model_metric&#39;: &#39;mixed_metric&#39;,<br> &#39;models&#39;: {&#39;30c8fc33166811cb906c1da65bb64634&#39;: {&#39;Model&#39;: &#39;AverageValueNaive&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;method&quot;: &quot;Mean&quot;,<br> &quot;window&quot;: null}&#39;, &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;mean&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;ClipOutliers&quot;, &quot;1&quot;: &quot;QuantileTransformer&quot;,<br> &quot;2&quot;: &quot;DifferencedTransformer&quot;},<br> &quot;transformation_params&quot;: {&quot;0&quot;: {&quot;method&quot;:<br> &quot;clip&quot;, &quot;std_threshold&quot;: 3, &quot;fillna&quot;: null},<br> &quot;1&quot;: {&quot;output_distribution&quot;: &quot;uniform&quot;,<br> &quot;n_quantiles&quot;: 44}, &quot;2&quot;: {}}}&#39;},<br> &#39;7a6ee57716e4a9f5766fdd360ca9cf90&#39;: {&#39;Model&#39;: &#39;AverageValueNaive&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;method&quot;: &quot;Mean&quot;,<br> &quot;window&quot;: null}&#39;, &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;akima&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;Detrend&quot;, &quot;1&quot;: &quot;bkfilter&quot;, &quot;2&quot;:<br> &quot;Slice&quot;}, &quot;transformation_params&quot;: {&quot;0&quot;:<br> {&quot;model&quot;: &quot;GLS&quot;, &quot;phi&quot;: 1, &quot;window&quot;:<br> 365}, &quot;1&quot;: {}, &quot;2&quot;: {&quot;method&quot;: 0.9}}}&#39;},<br> &#39;4686d91e12f95181e40aeb6cb7663e03&#39;: {&#39;Model&#39;: &#39;AverageValueNaive&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;method&quot;: &quot;Mean&quot;, &quot;window&quot;:<br> null}&#39;, &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;ffill_mean_biased&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;MinMaxScaler&quot;, &quot;1&quot;: &quot;ClipOutliers&quot;,<br> &quot;2&quot;: &quot;Detrend&quot;, &quot;3&quot;:<br> &quot;SeasonalDifference&quot;, &quot;4&quot;: &quot;Discretize&quot;,<br> &quot;5&quot;: &quot;ClipOutliers&quot;}, &quot;transformation_params&quot;:<br> {&quot;0&quot;: {}, &quot;1&quot;: {&quot;method&quot;: &quot;remove&quot;,<br> &quot;std_threshold&quot;: 4, &quot;fillna&quot;:<br> &quot;rolling_mean_24&quot;}, &quot;2&quot;: {&quot;model&quot;:<br> &quot;Tweedie&quot;, &quot;phi&quot;: 0.99, &quot;window&quot;: null},<br> &quot;3&quot;: {&quot;lag_1&quot;: 96, &quot;method&quot;: &quot;Mean&quot;},<br> &quot;4&quot;: {&quot;discretization&quot;: &quot;center&quot;,<br> &quot;n_bins&quot;: 5}, &quot;5&quot;: {&quot;method&quot;: &quot;clip&quot;,<br> &quot;std_threshold&quot;: 4, &quot;fillna&quot;: null}}}&#39;},<br> &#39;67ddeabd120f530563c94694364f513c&#39;: {&#39;Model&#39;: &#39;AverageValueNaive&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;method&quot;: &quot;Midhinge&quot;,<br> &quot;window&quot;: null}&#39;, &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;ffill&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;ClipOutliers&quot;, &quot;1&quot;: &quot;ClipOutliers&quot;,<br> &quot;2&quot;: &quot;Detrend&quot;, &quot;3&quot;: &quot;Discretize&quot;,<br> &quot;4&quot;: &quot;Discretize&quot;}, &quot;transformation_params&quot;:<br> {&quot;0&quot;: {&quot;method&quot;: &quot;remove&quot;, &quot;std_threshold&quot;:<br> 3.5, &quot;fillna&quot;: &quot;mean&quot;}, &quot;1&quot;:<br> {&quot;method&quot;: &quot;remove&quot;, &quot;std_threshold&quot;: 4,<br> &quot;fillna&quot;: &quot;rolling_mean_24&quot;}, &quot;2&quot;:<br> {&quot;model&quot;: &quot;Tweedie&quot;, &quot;phi&quot;: 0.99,<br> &quot;window&quot;: null}, &quot;3&quot;: {&quot;discretization&quot;:<br> &quot;center&quot;, &quot;n_bins&quot;: 20}, &quot;4&quot;:<br> {&quot;discretization&quot;: &quot;center&quot;, &quot;n_bins&quot;: 5}}}&#39;},<br> &#39;16bf50b53ca0e18367bd58e4794f309c&#39;: {&#39;Model&#39;: &#39;AverageValueNaive&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;method&quot;: &quot;Median&quot;,<br> &quot;window&quot;: null}&#39;, &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;median&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;PowerTransformer&quot;, &quot;1&quot;: &quot;QuantileTransformer&quot;,<br> &quot;2&quot;: &quot;QuantileTransformer&quot;, &quot;3&quot;:<br> &quot;Detrend&quot;}, &quot;transformation_params&quot;: {&quot;0&quot;: {},<br> &quot;1&quot;: {&quot;output_distribution&quot;: &quot;normal&quot;,<br> &quot;n_quantiles&quot;: 44}, &quot;2&quot;:<br> {&quot;output_distribution&quot;: &quot;uniform&quot;,<br> &quot;n_quantiles&quot;: 20}, &quot;3&quot;: {&quot;model&quot;:<br> &quot;Linear&quot;, &quot;phi&quot;: 1, &quot;window&quot;: null}}}&#39;}},<br> &#39;point_method&#39;: &#39;median&#39;, &#39;model_weights&#39;: {}}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5281380763532486</td>
</tr>
<tr>
<td>AutoTS(Theta)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> Ensemble<br><br> {}<br><br> {&#39;model_name&#39;: &#39;BestN&#39;, &#39;model_count&#39;: 1, &#39;model_metric&#39;: &#39;horizontal&#39;,<br> &#39;models&#39;: {&#39;f5a7c781ccf05947de1757470e82d165&#39;: {&#39;Model&#39;: &#39;Theta&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;deseasonalize&quot;: true, &quot;difference&quot;:<br> true, &quot;use_test&quot;: false, &quot;method&quot;: &quot;auto&quot;,<br> &quot;period&quot;: null, &quot;theta&quot;: 1.6, &quot;use_mle&quot;:<br> false}&#39;, &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;: &quot;ffill&quot;,<br> &quot;transformations&quot;: {&quot;0&quot;: &quot;ClipOutliers&quot;,<br> &quot;1&quot;: &quot;CumSumTransformer&quot;, &quot;2&quot;:<br> &quot;Slice&quot;, &quot;3&quot;: &quot;SeasonalDifference&quot;,<br> &quot;4&quot;: &quot;Discretize&quot;}, &quot;transformation_params&quot;:<br> {&quot;0&quot;: {&quot;method&quot;: &quot;remove&quot;,<br> &quot;std_threshold&quot;: 3, &quot;fillna&quot;: &quot;ffill&quot;},<br> &quot;1&quot;: {}, &quot;2&quot;: {&quot;method&quot;: 100}, &quot;3&quot;:<br> {&quot;lag_1&quot;: 28, &quot;method&quot;: &quot;Median&quot;},<br> &quot;4&quot;: {&quot;discretization&quot;: &quot;upper&quot;,<br> &quot;n_bins&quot;: 50}}}&#39;}}, &#39;model_weights&#39;: {}, &#39;point_method&#39;:<br> &#39;mean&#39;}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308699495773723</td>
</tr>
<tr>
<td>AutoTS(ARIMA)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> ARIMA<br><br> {&#39;fillna&#39;: &#39;akima&#39;, &#39;transformations&#39;: {&#39;0&#39;: &#39;ClipOutliers&#39;, &#39;1&#39;: &#39;Round&#39;,<br> &#39;2&#39;: &#39;Detrend&#39;}, &#39;transformation_params&#39;: {&#39;0&#39;: {&#39;method&#39;: &#39;remove&#39;,<br> &#39;std_threshold&#39;: 4, &#39;fillna&#39;: &#39;rolling_mean_24&#39;}, &#39;1&#39;: {&#39;decimals&#39;: 1,<br> &#39;on_transform&#39;: False, &#39;on_inverse&#39;: True}, &#39;2&#39;: {&#39;model&#39;: &#39;GLS&#39;, &#39;phi&#39;: 1,<br> &#39;window&#39;: 10}}}<br><br> {&#39;p&#39;: 3, &#39;d&#39;: 1, &#39;q&#39;: 1, &#39;regression_type&#39;: None}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308641975308641</td>
</tr>
<tr>
<td>AutoTS(ARDL)</td>
<td>AutoTS + statsmodels</td>
<td>Initiated<br> AutoTS object with best model: <br><br> Ensemble<br><br> {}<br><br> {&#39;model_name&#39;: &#39;BestN&#39;, &#39;model_count&#39;: 3, &#39;model_metric&#39;: &#39;best_score&#39;,<br> &#39;models&#39;: {&#39;32b7702842dcb678cb7cfb82995d23c2&#39;: {&#39;Model&#39;: &#39;ARDL&#39;,<br> &#39;ModelParameters&#39;: &#39;{&quot;lags&quot;: 3, &quot;trend&quot;: &quot;ct&quot;,<br> &quot;order&quot;: 0, &quot;regression_type&quot;: &quot;holiday&quot;}&#39;,<br> &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;rolling_mean_24&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;MinMaxScaler&quot;, &quot;1&quot;: &quot;QuantileTransformer&quot;,<br> &quot;2&quot;: &quot;ScipyFilter&quot;}, &quot;transformation_params&quot;:<br> {&quot;0&quot;: {}, &quot;1&quot;: {&quot;output_distribution&quot;:<br> &quot;uniform&quot;, &quot;n_quantiles&quot;: 44}, &quot;2&quot;:<br> {&quot;method&quot;: &quot;butter&quot;, &quot;method_args&quot;: [2, 0.77,<br> &quot;lowpass&quot;, false]}}}&#39;}, &#39;49e498ee9548082bb65cc5dd4f1ab3a8&#39;:<br> {&#39;Model&#39;: &#39;ARDL&#39;, &#39;ModelParameters&#39;: &#39;{&quot;lags&quot;: 3,<br> &quot;trend&quot;: &quot;ct&quot;, &quot;order&quot;: 0,<br> &quot;regression_type&quot;: &quot;holiday&quot;}&#39;,<br> &#39;TransformationParameters&#39;: &#39;{&quot;fillna&quot;:<br> &quot;rolling_mean_24&quot;, &quot;transformations&quot;: {&quot;0&quot;:<br> &quot;MinMaxScaler&quot;, &quot;1&quot;: &quot;QuantileTransformer&quot;,<br> &quot;2&quot;: &quot;ScipyFilter&quot;, &quot;3&quot;: &quot;Detrend&quot;},<br> &quot;transformation_params&quot;: {&quot;0&quot;: {}, &quot;1&quot;:<br> {&quot;output_distribution&quot;: &quot;uniform&quot;,<br> &quot;n_quantiles&quot;: 44}, &quot;2&quot;: {&quot;method&quot;:<br> &quot;butter&quot;, &quot;method_args&quot;: [2, 0.77, &quot;lowpass&quot;,<br> false]}, &quot;3&quot;: {&quot;model&quot;: &quot;Linear&quot;,<br> &quot;phi&quot;: 0.999, &quot;window&quot;: null}}}&#39;},<br> &#39;b1647f7d0e7c66b2e952fc33087d9d56&#39;: {&#39;Model&#39;: &#39;ARDL&#39;, &#39;ModelParameters&#39;:<br> &#39;{&quot;lags&quot;: 1, &quot;trend&quot;: &quot;ct&quot;, &quot;order&quot;:<br> 0, &quot;regression_type&quot;: null}&#39;, &#39;TransformationParameters&#39;:<br> &#39;{&quot;fillna&quot;: &quot;rolling_mean_24&quot;, &quot;transformations&quot;:<br> {&quot;0&quot;: &quot;MinMaxScaler&quot;, &quot;1&quot;:<br> &quot;QuantileTransformer&quot;, &quot;2&quot;: &quot;ScipyFilter&quot;,<br> &quot;3&quot;: &quot;Detrend&quot;}, &quot;transformation_params&quot;:<br> {&quot;0&quot;: {}, &quot;1&quot;: {&quot;output_distribution&quot;:<br> &quot;uniform&quot;, &quot;n_quantiles&quot;: 44}, &quot;2&quot;:<br> {&quot;method&quot;: &quot;butter&quot;, &quot;method_args&quot;: [2, 0.77,<br> &quot;lowpass&quot;, false]}, &quot;3&quot;: {&quot;model&quot;: &quot;Linear&quot;,<br> &quot;phi&quot;: 0.999, &quot;window&quot;: null}}}&#39;}}, &#39;model_weights&#39;: {},<br> &#39;point_method&#39;: &#39;mean&#39;}<br><br> SMAPE: 9.876543209876543<br><br> MAE: 0.04938271604938271<br><br> SPL: 1.5308641975308641</td>
</tr>
</tbody>
</table>
"""

#formula used
with Formula_us:
     st.title("The Theta Model(statmodels) ")
     st.markdown("For the development of the model, several models were evaluated to identify in which sites or warehouses there could be depleted elements in the inventories of the warehouses, the theta model was selected in the one implemented in the package [statsmodels](https://www.statsmodels.org/devel/index.html) this module is consumed in the package [AutoTS](https://github.com/winedarksea/AutoTS) which is a module for the modeling of time series, this module allows to choose manually or automatically models that allow to establish predictions based on time or date, the models tested are briefly described below. Their results are very similar, for this reason the Theta model was selected, since it presents clearer variations in the predictions.")
     st.markdown("A straightforward forecasting technique is the Theta model of Assimakopoulos and Nikolopoulos (2000), which entails fitting two lines, smoothing the lines with a Simple Exponential Smoother, and then combining the forecasts from the two lines to produce the final forecast.")
     st.markdown(" for more information [model theta](https://www.statsmodels.org/devel/examples/notebooks/generated/theta-model.html)")
     st.markdown("The following table shows the different types of models reviewed and their input parameters and the following metric values :")
     st.markdown("SMAPE is Symmetric Mean Absolute Percentage Loss and is generally the most versatile metric across multiple series as it is scaled")
     st.markdown("MAE, RMSE, MAGE, MLE, iMLE are unscaled and accordingly in multivariate forecasting will favor model performance on the largest scale input series")
     st.markdown("SPL is Scaled Pinball Loss, sometimes called Quantile Loss, and is the optimal metric for optimizing upper/lower quantile forecast accuracies.")
     st.markdown(table_models,unsafe_allow_html=True)

#data in
with Data__in:
     st.title("Data used ")
     st.markdown("The data used is a union between the dataset of products, points of sale and out of stock items in the inventory, to make the model more functional filters are added by department, city, sales chain or brand, product category and subcategory or the specific product.")






st.set_option('deprecation.showPyplotGlobalUse', False)

#sidebar
st.sidebar.markdown("Developed by team 108 :globe_with_meridians: for DS4A Colombia cohort 6.")
st.sidebar.write(f'''
    <a target="_blank" href="https://main.d1apizaihpmcr4.amplifyapp.com/">
        <button>
            Return to project page
        </button>
    </a>
    ''',
    unsafe_allow_html=True
)
st.sidebar.markdown(" &copy; 2022 &copy;")


#Adding datasets
#url_5 = 'https://eficaciadata.s3.amazonaws.com/ago_pdv_pro.csv' # Data of products out stock for store and type of product
#url_5 = 'csv/ago_pdv_pro.csv' # Data of products out stock for store and type of product
#loading data
#data_5 = pd.read_csv(url_5)

#parquets

url_5 = 'parquet/ago_pdv_pro.parquet' # Data of products out stock for store and type of product

#uses this instruccion it the data change @st.cache(persist=True)( If you have a different use case where the data does not change so very often, you can simply use this)


#loading data

data_5 = pd.read_parquet(url_5, engine='auto')


#SUBMENUS
ANTIOQUIA = 'Medellin','Caucasia','El Bagre','Zaragoza','La Ceja','Rionegro','Bello','Copacabana','Marinilla','Itagui','Envigado','Caldas','Sabaneta','Turbo','Apartadó','Carepa','La Estrella','San Jerónimo','Barbosa','Fredonia','Amagá','Necoclí','La Unión','Carmen De Viboral','Retiro'
#ARAUCA = "Arauca"
ATLANTICO = 'Barranquilla','Soledad','Malambo','Sabanalarga','Candelaria','Galapa','Puerto Colombia','Santo Tomás','Sabanagrande','Palmar De Varela','Baranoa'
#BOGOTADC = "Bogotá"
BOLIVAR = 'Cartagena','Magangue','Arjona','Turbaco','San Juan Nepomuceno','Mompos','Carmen De Bolívar'
BOYACA ='Tunja','Duitama','Sogamoso','Paipa','Chiquinquirá','Villa De Leyva','Samacá','Puerto Boyacá'
CALDAS = 'Manizales','La Dorada','Villamaría','Chinchiná'
#CAQUETA= 'Florencia'
CASANARE = 'Yopal'
CAUCA = 'Popayán','Santander De Quilichao','Puerto Tejada'
CESAR = 'Valledupar','Aguachica','Bosconia','Curumaní'
CHOCO= 'Quibdó'
CORDOBA = 'Montería','Cereté','La Apartada','Lorica','Planeta Rica','Sahagún','Ciénaga De Oro','Tierralta'
CUNDINAMARCA= 'Chía','Girardot','Facatativá','Mosquera','Soacha','Zipaquirá','Madrid','Cajicá','Fusagasugá','La Calera','Ricaurte','Cota','La Mesa','Villeta','Villa De San Diego De Ubate','Chocontá','Funza','Gachancipá','Pacho','Tocancipá','Villa Pinzón','Tocaima'
HUILA = 'Neiva','Pitalito','San Agustín','Campoalegre'
LAGUAJIRA = 'Riohacha','Albania','Fonseca','Maicao','San Juan Del Cesar'
MAGDALENA= 'Santa Marta','Aracataca','Pivijay','El Banco','Ciénaga','Fundación'
META = 'Villavicencio','Restrepo','Acacías'
NARINO = 'Pasto','Ipiales'
NORTEDESANTANDER = 'Cúcuta','Ocaña'
QUINDIO = 'Armenia','Calarca','Montenegro','Quimbaya','La Tebaida','Circasia'
RISARALDA = 'Pereira','Dosquebradas','Santa Rosa De Cabal'
SANTANDER = 'Floridablanca','Bucaramanga','Barrancabermeja','Piedecuesta','San Gil','Cerrito','San Miguel'
SUCRE ='Sincelejo','Corozal','Toluviejo','Coveñas','Sampués','San Onofre'
TOLIMA ='Flandes','Ibagué','Espinal','Melgar'
VALLEDELCAUCA ='Santiago De Cali','Palmira','Jamundí','Guadalajara De Buga','Cartago','Tulúa','Buenaventura','Guacarí','Caicedonia','Zarzal','Yumbo','Bugalagrande','Ginebra','Roldanillo','Sevilla'


#formula used
with submenus:
     st.title("Select options for the model")
     select_depto = st.selectbox('Select a department of Colombia',
                                ('ANTIOQUIA','ARAUCA','ATLÁNTICO',
                                 'BOGOTA D.C','BOLIVAR','BOYACA',
                                 'CALDAS','CAQUETA','CASANARE',
                                 'CAUCA','CESAR','CHOCO','CORDOBA',
                                 'CUNDINAMARCA','HUILA','LA GUAJIRA',
                                 'MAGDALENA','META','NARIÑO','NORTE DE SANTANDER',
                                 'QUINDÍO','RISARALDA','SANTANDER','SUCRE','TOLIMA','VALLE DEL CAUCA'))

     if select_depto == 'ANTIOQUIA' :
        select_ciudad = st.selectbox(
        'Select a city',(ANTIOQUIA))
     elif select_depto == 'ARAUCA' :
         select_ciudad = 'Arauca'
     elif select_depto == 'ATLÁNTICO':
         select_ciudad = st.selectbox(
         'Select a city',(ATLANTICO))
     elif select_depto == 'BOGOTA D.C' :
         select_ciudad="Bogotá"
     elif select_depto == 'BOLIVAR' :
         select_ciudad = st.selectbox(
         'Select a city',(BOLIVAR))
     elif select_depto == 'BOYACA' :
         select_ciudad = st.selectbox(
         'Select a city',(BOYACA))
     elif select_depto == 'CALDAS' :
         select_ciudad = st.selectbox(
         'Select a city',(CALDAS))
     elif select_depto == 'CAQUETA' :
         select_ciudad="Florencia"
     elif select_depto == 'CASANARE' :
         select_ciudad = st.selectbox(
         'Select a city',"Yopal")
     elif select_depto == 'CAUCA' :
         select_ciudad = st.selectbox(
         'Select a city',(CAUCA))
     elif select_depto == 'CESAR' :
         select_ciudad = st.selectbox(
         'Select a city',(CESAR))
     elif select_depto == 'CHOCO' :
         select_ciudad = st.selectbox(
         'Select a city',(CHOCO))
     elif select_depto == 'CORDOBA' :
         select_ciudad = st.selectbox(
         'Select a city',(CORDOBA))
     elif select_depto == 'CUNDINAMARCA' :
         select_ciudad = st.selectbox(
         'Select a city',(CUNDINAMARCA))
     elif select_depto == 'LA GUAJIRA' :
         select_ciudad = st.selectbox(
         'Select a city',(LAGUAJIRA))
     elif select_depto == 'MAGDALENA' :
         select_ciudad = st.selectbox(
         'Select a city',(MAGDALENA))
     elif select_depto == 'META' :
         select_ciudad = st.selectbox(
         'Select a city',(META))
     elif select_depto == 'NARIÑO' :
         select_ciudad = st.selectbox(
         'Select a city',(NARINO))
     elif select_depto == 'NORTE DE SANTANDER' :
         select_ciudad = st.selectbox(
         'Select a city',(NORTEDESANTANDER))
     elif select_depto == 'QUINDÍO' :
         select_ciudad = st.selectbox(
         'Select a city',(QUINDIO))
     elif select_depto == 'RISARALDA' :
         select_ciudad = st.selectbox(
         'Select a city',(RISARALDA))
     elif select_depto == 'SANTANDER':
         select_ciudad = st.selectbox(
         'Select a city',(SANTANDER))
     elif select_depto == 'SUCRE' :
         select_ciudad = st.selectbox(
         'Select a city',(SUCRE))
     elif select_depto == 'TOLIMA' :
         select_ciudad = st.selectbox(
         'Select a city',(TOLIMA))
     elif select_depto == 'VALLE DEL CAUCA' :
         select_ciudad = st.selectbox(
         'Select a city',(VALLEDELCAUCA))

     data_5_fil= data_5.loc[data_5['Ciudad'] == select_ciudad]

     select_cadena = st.selectbox('Select a brand of stores for this city', options=pd.unique(data_5_fil['Cadena']))
     data_5_cad = data_5_fil.loc[data_5_fil['Cadena'] == select_cadena]
     select_categoria = st.selectbox('Select a product category',	options=pd.unique(data_5_cad['Categoria']))
     data_5_cat = data_5_cad.loc[data_5_cad['Categoria'] == select_categoria]
     select_subcategoria = st.selectbox('Select a sub-category',	options=pd.unique(data_5_cat['SubCategoria']))
     data_5_subcat = data_5_cat.loc[data_5_cat['SubCategoria'] == select_subcategoria]
    # demasido complicado el manejo de errores try select_subcategoria == False:
    #   except ValueError:
#st.error(":cold_sweat::cold_sweat:The combination of selections applied does not generate optimal results, select other options. not all products are available in all stores. ")






#model explanation
with Model_exp:
    st.title("Time series using AutoTS")
    st.markdown("WHY")
    data_5_subcat.Agotado = data_5_subcat.Agotado.replace({True: 1, False: 0})
    st.write(data_5_subcat.head(3))
    #creating modeles, autoTS tries and get the best model
    #model_list = ['LastValueNaive', 'GLS', 'GLM', 'ETS', 'AverageValueNaive', 'ARIMA', 'Theta', 'ARDL']
    model_list = ['Theta']
    #for performance the max values of tries are 1 and a 1 validation(zero value makes one validation), predidtions for 60 days the frecuency its automatic selected with the data
    model = AutoTS(forecast_length=100, frequency='infer', prediction_interval=0.80, ensemble='simple', model_list=model_list, transformer_list='all', max_generations=2, num_validations=0)
    model = model.fit(data_5_subcat, date_col='Fecha', value_col='Agotado', id_col=None)
    prediction = model.predict()
    forecast = prediction.forecast
    # Print the description of the best model
    st.markdown("following thr options of the best model selected")
    st.write(model)
    st.markdown("Out Stock Prediction")
    st.write(forecast)





#graph of the model
with ga_model:
     st.title("Graph of results ")
     st.markdown("Explanation ...")


     st.subheader('Prediction Chart model Theta')

     fig_m2 = prediction.plot(model.df_wide_numeric, remove_zeroes=False)
     st.pyplot(fig_m2)

     st.subheader('Iterations Chart of model Theta' )
     fig_m3 = model.plot_generation_loss()
     st.pyplot(fig_m3)













