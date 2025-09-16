# Supervised Learning (II): Regression



:::{objectives}
- Understand the fundamental concept of regression (overfitting, cross-validation, gradient search, ...)
- Distinguish between types of regression: simple vs. multiple regression, linear vs. non-linear regression, and other specialized regression methods.
- Perform regression tasks using representative algorithms (*e.g.*, k-NN, Linear Regression, Polynomial Regression, Support Vector Regression, Decision Tree, and Multi-Layer Perceptron)
- Evaluate model performance with metrics such as Root Mean Squared Error (RMSE) and the R-squared (R²) score, and visualize predictive curves.
:::



:::{instructor-note}
- 40 min teaching/demonstration
- 40 min exercises
:::



## Regression


Regression is a type of supervised machine learning task where the goal is to predict a continuous numerical value based on input features. Unlike classification, which assigns outputs to discrete categories, regression models produce real-valued predictions.

Although the Penguins dataset is most commonly used for classification tasks, it can also be applied to regression problems by choosing a continuous target variable. From the pairplot, we can observe a strong visual relationship between body mass and flipper length, indicating a clear positive correlation. Consequently, we select these two features for the regression task, aiming to estimate body mass based on flipper length.

:::{figure} ./images/4-penguins-pairplot.png
:align: center
:width: 100%
:::


Depending on the model construction approach, in this episode we explore a variety of regression algorithms to predict penguin body mass based on flipper length. These models are selected to represent different categories of machine learning approaches, ranging from simple, interpretable methods to more complex, flexible ones.
- **KNN Regression**: Predictions are made based on the average of the closest training samples. This non-parametric, instance-based model captures local patterns in the data effectively.
- **Linear Models**: Standard Linear Regression and Regularized Regression assume a straight-line relationship between flipper length and body mass. These models are interpretable and efficient, providing a solid baseline for comparison.
- **Non-linear Models**: To account for possible non-linear trends, we include Polynomial Regression with higher-degree terms and Support Vector Regression (SVR) with ``rbf`` kernels, which can model more complex relationships.
- **Tree-based Models**: Decision Trees, Random Forests, and Gradient Boosting offer robust alternatives by recursively partitioning the feature space or combining ensembles to improve accuracy and handle non-linearities effectively.
- **Neural Networks**: These serve as universal function approximators, capable of learning intricate patterns in the data, but typically require larger datasets and more computational resources.


Each model’s performance is rigorously assessed using cross-validated metrics such as Root Mean Squared Error (RMSE) and R². The resulting predictive curves illustrate how well each model captures the biological relationship between flipper length and body mass.



## Data Preparation


Similar to the procedures adopted in previous episodes, we follow the same preprocessing steps for the Penguins dataset, including handling missing values and detecting outliers. For the regression task, categorical features are not needed, so encoding them is unnecessary.



## Data Processing


Below is the code script to extract ``flipper_length_mm`` and ``body_mass_g`` features from the main dataset.
```python
X = penguins_regression[["flipper_length_mm"]].values
y = penguins_regression["body_mass_g"].values
```

In this episode, we first perform feature scaling, followed by splitting the data into training and testing sets. The ``inverse_transform()`` method reverts transformed data back to its original scale or format.

```python
from sklearn.preprocessing import StandardScaler
   
# standardize feature and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=123)

X_train_orig = scaler_X.inverse_transform(X_train_scaled).ravel()
y_train_orig = scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).ravel()
...
```

:::{figure} ./images/6-spliting-training-testing-dataset.png
:align: center
:width: 80%
:::



## Training Model & Evaluating Model Performance



### k-Nearest Neighbors (KNN)


We begin by applying the KNN algorithm to the penguin regression task, as illustrated in the code example below.
```python
from sklearn.neighbors import KNeighborsRegressor

knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train_scaled, y_train_scaled)

# predict on test data
y_pred_knn_scaled = knn_model.predict(X_test_scaled)
y_pred_knn = scaler_y.inverse_transform(y_pred_knn_scaled.reshape(-1, 1)).ravel()
```

For the regression task, we use Root Mean Squared Error (RMSE) and R² score as evaluation metrics.
- RMSE measures the average magnitude of prediction errors, providing insight into how closely the model’s predictions match the actual values
- R² score indicates the proportion of variance in the target variable that is explained by the model, reflecting its overall goodness of fit.

```python
# evaluate model performance
from sklearn.metrics import root_mean_squared_error, r2_score

rmse_knn = root_mean_squared_error(y_test_orig, y_pred_knn)
r2_value_knn = r2_score(y_test_orig, y_pred_knn)
print(f"K-Nearest Neighbors RMSE: {rmse_knn:.2f}, R²: {r2_value_knn:.2f}")
```


To visualize the KNN algorithm for the regression task, we plot the **predictive curve**, which maps input values to predicted outputs. This curve illustrates how KNN responds to changes in a single feature. Since KNN is a non-parametric, instance-based method, it does not learn a fixed equation during training. Instead, predictions are made by averaging the target values of the *k* nearest training examples for each input.

The resulting predictive curve is typically piecewise-smooth, adapting to local patterns in the data. That is, the curve may bend or flatten depending on regions where data points are dense or sparse.

:::{figure} ./images/6-regression-predictive-curve-knn-5.png
:align: center
:width: 80%
:::

This makes the predictive curve an especially useful tool for assessing whether KNN is underfitting (*e.g.*, when *k* is large) or overfitting (e.g., when *k* is small). By adjusting *k* and observing changes in the curve’s shape, we can intuitively tune the model’s **bias-variance tradeoff**.

:::{figure} ./images/6-regression-predictive-curve-knn-1357.png
:align: center
:width: 80%
:::

:::{callout} The bias-variance tradeoff

The bias-variance tradeoff is a fundamental concept in machine learning that describes the balance between model simplicity and model flexibility when trying to make accurate predictions.
- Bias measures how much a model’s predictions systematically differ from the true values. High bias means the model is too simple and cannot capture the underlying patterns in the data, which leads to underfitting.
- Variance measures how much a model’s predictions change when trained on different datasets. High variance means the model is too sensitive to small fluctuations in the training data, which leads to overfitting.
:::



### Linear Regression


Having explored a KNN regressor to predict penguin body mass from flipper length, we now turn to a fundamental and interpretable alternative: the Linear Regression model. While KNN makes predictions based on the average mass of the most similar observations, linear regression aims to identify a single, global linear relationship between the two variables. This approach fits a straight line through the data that minimizes the overall prediction error, producing a model that is typically less computationally intensive and offers immediate insight into the underlying trend.

The core concept of this linear model is a simple equation:
:::{math}
body\_mass = \beta_0 + \beta_1 × flipper\_length
:::
- the coefficient, $\beta_1$, represents the model’s estimate of how much a penguin’s body mass increases for each additional millimeter of flipper length
- the intercept, $\beta_0$, indicates the theoretical body mass for a penguin with a flipper length of zero. While this value is not biologically meaningful, it is necessary to position the line correctly.

The fitted values of $\beta_1$ and $\beta_0$ can be accessed via ``model.coef_`` and ``model.intercept_``, respectively. ​This equation provides a direct and interpretable rule: for any given flipper length, we can calculate a precise predicted body mass with given $\beta_1$ and $\beta_0$.

```python
from sklearn.linear_model import LinearRegression

linear_model = LinearRegression()
linear_model.fit(X_train_scaled, y_train_scaled)
print(linear_model.coef_, linear_model.intercept_)

y_pred_linear_scaled = linear_model.predict(X_test_scaled)
y_pred_linear = scaler_y.inverse_transform(y_pred_linear_scaled.reshape(-1, 1)).ravel()
```

Once trained, we evaluate the linear regression model’s predictive performance on the testing set using the same metrics: RMSE and R² score. In the Penguins dataset, a high R² indicates that flipper length is a strong predictor of body mass, while a low RMSE reflects precise predictions. These metrics also allow for direct comparison with KNN and other models, such as Polynomial Regression and tree-based methods that will be discussed below, highlighting situations where the simple linear assumption is sufficient and where it may fall short.

```python
rmse_linear = root_mean_squared_error(y_test_orig, y_pred_linear)
r2_value_linear = r2_score(y_test_orig, y_pred_linear)
print(f"Linear Regression RMSE: {rmse_linear:.2f}, R²: {r2_value_linear:.2f}")
```

The resulting predictive curve is shown below.

:::{figure} ./images/6-regression-predictive-curve-linear.png
:align: center
:width: 80%
:::


**Residual analysis**

While metrics like RMSE and R-squared scores provide a high-level summary of model performance, **residual analysis** allows us to examine the model more deeply and verify the key assumptions of linear regression, ensuring that its conclusions are valid and reliable. Residuals are the differences between the observed body mass values and the values predicted by the model.

From the figure below, we can see that the residuals are randomly scattered around zero, with no apparent systematic patterns. This indicates that the linear model is largely unbiased and effectively captures the main trend between flipper length and body mass.

:::{figure} ./images/6-regression-linear-residual-analysis.png
:align: center
:width: 100%
:::


:::{note}
If we notice certain patterns, *i.e.*, residuals that consistently increase or decrease with larger flipper lengths, it suggests that the relationship between body mass and flipper length might not be purely linear. Similarly, if the residuals fan out, showing greater spread at higher predicted values, it indicates heteroscedasticity, meaning the model errors are not consistent across the range of predictions. Such patterns imply that a simple linear regression model may not fully capture the variability in body mass.
:::


Another key aspect of residual analysis involves assessing normality, as linear regression assumes normally distributed residuals for reliable inference. For the Penguins dataset, this can be evaluated using a histogram or a Q-Q (quantile-quantile) plot of the residuals.

The histogram of residuals illustrates the distribution of prediction errors across the dataset. In the Penguins dataset, these residuals should form a roughly symmetric, bell-shaped curve centered at zero. This indicates that the model is not systematically over-predicting or under-predicting body mass, and that most errors are relatively small, with fewer large deviations.

The Q-Q plot compares the distribution of the residuals to a theoretical normal distribution. On the plot, the x-axis represents the expected quantiles from a standard normal distribution, while the y-axis shows the quantiles of the observed residuals. If the residuals are normally distributed, the points should align closely with the diagonal reference line.


**Overfitting and underfitting**

In the previous section, we evaluated the Linear Regression model on the testing dataset and calculated metrics such as RMSE and R² to understand its predictive performance. While this gives a good indication of how well this model generalizes to unseen data, it only tells half the story. 

To get a complete picture, it is important to also assess this model’s performance on the training dataset and compare it with the testing results. This comparison is the primary diagnostic tool for identifying a model's fundamental flaw: whether it is learning the underlying signal or merely memorizing the data.

By calculating performance metrics like RMSE and R-squared for both training and testing datasets, we can identify potential issues such as overfitting and underfitting.
- **Overfitting** occurs when the model performs extremely well on the training data but poorly on the testing data. This indicates that the model has memorized the training patterns, including noise, rather than capturing the true underlying relationship.
- **Underfitting** happens when the model performs poorly on both training and testing datasets, suggesting that it is too simple to capture the relevant trends in the data.

```python
# --- Training data predictions ---
y_pred_train_scaled = linear_model.predict(X_train_scaled)
y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1, 1)).ravel()

rmse_linear_train = root_mean_squared_error(y_train_orig, y_pred_train)
r2_linear_train = r2_score(y_train_orig, y_pred_train)

print(f"Linear Regression (Train) RMSE: {rmse_linear_train:.2f}, R²: {r2_linear_train:.2f}")

# --- Testing data predictions ---
y_pred_test_scaled = linear_model.predict(X_test_scaled)
y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1, 1)).ravel()

rmse_linear_test = root_mean_squared_error(y_test_orig, y_pred_test)
r2_linear_test = r2_score(y_test_orig, y_pred_test)

print(f"Linear Regression (Test)  RMSE: {rmse_linear_test:.2f}, R²: {r2_linear_test:.2f}")

# Linear Regression (Train) RMSE: 387.10, R²: 0.77
# Linear Regression (Test)  RMSE: 411.85, R²: 0.72
```

The trained Linear Regression model for predicting penguin body mass based on flipper length in the penguins dataset achieves comparable RMSE and R² scores on both the training and testing datasets, indicating a fairly good model.

::::::{exercise} Regularized Regression (Ridge and Lasso)

To address overfitting and underfitting, regularized regression methods, such as **Ridge** and **Lasso** regression, extend linear regression by adding a penalty term to the standard cost function. This penalty discourages the model from relying too heavily on any single feature or from becoming overly complex by forcing coefficient values to be small.
- **Ridge Regression** (L2 regularization) shrinks coefficients towards zero but never entirely eliminates them, which is highly effective for handling correlated features and improving stability. This is common in the penguins dataset when predictors like flipper length and bill length are correlated. 
- **Lasso Regression** (L1 regularization) can drive some coefficients to exactly zero, effectively performing automatic feature selection and creating a simpler, more interpretable model. For instance, Lasso might retain flipper length while discarding less predictive features, improving generalization.

In this exercise (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/6-ML-Regression.ipynb)), we will
- Train the Penguins dataset using Ridge and Lasso regression models, and compare their fitted parameters, RMSE, and R² scores.
- Conduct a residual analysis to evaluate whether the regularized regression models achieve better performance than the standard linear regression model.


:::::{tabs}
::::{group-tab} Ridge Regression

```python
from sklearn.linear_model import Ridge

ridge_model = Ridge(alpha=20.0)
ridge_model.fit(X_train_scaled, y_train_scaled)
print(ridge_model.coef_, ridge_model.intercept_)

y_pred_ridge_scaled = ridge_model.predict(X_test_scaled)
y_pred_ridge = scaler_y.inverse_transform(y_pred_ridge_scaled.reshape(-1, 1)).ravel()

rmse_ridge = root_mean_squared_error(y_test_orig, y_pred_ridge)
r2_value_ridge = r2_score(y_test_orig, y_pred_ridge)
print(f"Regularized Regression (Ridge) RMSE: {rmse_ridge:.2f}, R²: {r2_value_ridge:.2f}")
```
::::

::::{group-tab} Lasso Regression
```python
from sklearn.linear_model import Lasso

lasso_model = Lasso(alpha=0.2)
lasso_model.fit(X_train_scaled, y_train_scaled)
print(lasso_model.coef_, lasso_model.intercept_)

y_pred_lasso_scaled = lasso_model.predict(X_test_scaled)
y_pred_lasso = scaler_y.inverse_transform(y_pred_lasso_scaled.reshape(-1, 1)).ravel()

rmse_lasso = root_mean_squared_error(y_test_orig, y_pred_lasso)
r2_value_lasso = r2_score(y_test_orig, y_pred_lasso)
print(f"Regularized Regression (Lasso) RMSE: {rmse_lasso:.2f}, R²: {r2_value_lasso:.2f}")
```
::::

:::{figure} ./images/6-regression-predictive-curve-linear-ridge-lasso.png
:align: center
:width: 80%
:::

:::::
::::::


### Polynomial Regression

In the previous section, we assumed that penguin body mass is linearly proportional to flipper length, and after training, we have verified that this assumption holds reasonably well. However, for other applications, if two variables are explicitly not linearly related, and a simple linear model may fail to capture the underlying patterns. In such cases, we can resort to polynomial regression to capture non-linear relationship by including higher-degree terms of the predictor variable.

In the context of the Penguins dataset, polynomial regression extends linear regression by modeling body mass as a polynomial function of flipper length with the formula as
:::{math}
body\_mass = \beta_0 + \beta_1 × flipper\_length + \beta_2 × flipper\_length^2 + \beta_3 × flipper\_length^3 + ...
:::

This approach allows the model to fit a curved relationship, which might be relevant if, for example, body mass increases more rapidly with flipper length for larger penguins, as seen in species like Gentoo.

The process of training a Polynomial Regression model is similar to Linear Regression. We first transform the original feature (flipper length) by adding polynomial terms (*e.g.*, $flipper\_length^2$ and higher-degree terms), creating a feature matrix that the Polynomial Regression model uses to fit a non-linear curve while still employing Linear Regression techniques on the transformed features.

```python
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degree3=3
poly3_model = make_pipeline(PolynomialFeatures(degree3), LinearRegression())
poly3_model.fit(X_train_scaled, y_train_scaled)
print(poly3_model.named_steps['linearregression'].coef_, poly3_model.named_steps['linearregression'].intercept_)

y_pred_poly3_scaled = poly3_model.predict(X_test_scaled)
y_pred_poly3 = scaler_y.inverse_transform(y_pred_poly3_scaled.reshape(-1, 1)).ravel()

rmse_poly3 = root_mean_squared_error(y_test_orig, y_pred_poly3)
r2_value_poly3 = r2_score(y_test_orig, y_pred_poly3)
print(f"Polynomial Regression (degree={degree3}) RMSE: {rmse_poly3:.2f}, R²: {r2_value_poly3:.2f}")
```

The Polynomial Regression model is trained by minimizing the sum of squared and cubic residuals, and performance is evaluated using metrics like RMSE and R². Compared with Linear Regression, we see that a third-degree polynomial regression model provides a marginally better fit than the simple linear model.


Below we present the predictive curves for Polynomial Regression models with degrees 3 and 5, alongside the curve for Linear Regression. In addition, we report the evaluation metrics (RMSE and R²) on the testing dataset to provide a quantitative comparison.


:::{figure} ./images/6-regression-predictive-curve-linear-poly35.png
:align: center
:width: 80%
:::


:::{table} Performance metrics (RMSE and R²) on the testing dataset
:align: center
:widths: auto

| Model | RMSE | R²  |
| :---: | :--: | :-: |
| Linear Regression | 411.85 | 0.72 |
| Ridge Regression | 414.18 | 0.72 |
| Lasso Regression | 437.19 | 0.69 |
| Polynomial Regression <br>(degree=3) | 407.47 | 0.73 |
| Polynomial Regression <br>(degree=5) | 415.55 | 0.72 |
| Support Vector Regression | 424.24 | 0.70 |
:::


:::{caution}
It is crucial to approach this added complexity with caution. While higher-degree polynomials can achieve very close fits to the training data, they are also highly prone to overfitting. For example, a model with a very high degree (*e.g.*, degree = 10) may contort itself to pass through nearly every training point, capturing random noise rather than the true underlying biological relationship. As a result, such a model would likely perform poorly on unseen test data, sacrificing generalizability for apparent short-term accuracy.
:::


Additionally, residual analysis can provide further information on the model performance. 


:::{figure} ./images/6-regression-linear-poly5-residual-analysis.png
:align: center
:width: 100%
:::


Compared with Linear Regression, the Polynomial Regression model with degree = 5 shows signs of overfitting, as evidenced by systematic deviations in the residuals and the asymmetric distribution of prediction errors across the dataset.


Overall, Polynomial Regression serves as a simple yet powerful extension of Linear Regression. It enables us to capture non-linear relationships while still benefiting from the interpretability and computational efficiency of a linear framework applied to transformed features.
The key challenge lies in selecting the appropriate polynomial degree. Too low a degree may underfit the data, missing important trends, while too high a degree risks memorizing noise and overfitting. To strike the right balance, rigorous evaluation techniques --- such as cross-validation on the training set --- are typically used to identify the optimal degree, followed by a final assessment on the testing set. By carefully tuning complexity, polynomial regression can deliver genuine improvements in predictive accuracy and provide a more faithful representation of the often non-linear patterns found in the natural world.



### Support Vector Machine


In the previous episode, we introduced the SVM model, which is widely recognized for its effectiveness in classification tasks by finding an optimal hyperplane that maximizes the margin between classes. Here, we adapt the same principles to regression through **Support Vector Regression** (SVR). Unlike Linear Regression or Polynomial Regression, which minimize squared errors, SVR builds on the concepts of margins and support vectors, and aims to find a tube (or a channel) (ε-insensitive zone) of a specified width that captures as many data points as possible, while only the points lying outside this tube (the support vector) affect the model’s predictions.


The core challenge SVR faces is that, by its fundamental nature, it seeks a linear relationship (a flat hyperplane). In many real-world problems, such as predicting a penguin’s body mass from its flipper length, the underlying relationship, while roughly linear, may contain subtle non-linear patterns that a straight line cannot fully capture.

Rather than manually generating polynomial features, which can be computationally expensive and impractical in high-dimensional spaces, kernel functions are used to capture non-linear relationships by implicitly projecting (rather than explicitly transforming) the input data into higher-dimensional feature spaces.


:::{note}
Several kernel types are commonly used in SVR, each imparting different characteristics to the model:
- **Linear Kernel**: The simplest kernel, which does not perform any transformation and assumes a linear relationship between features and the target variable. It is fast and interpretable but lacks flexibility for modeling complex patterns.
- **Polynomial Kernel**: This kernel enables the model to fit polynomial curves of a specified degree $d$ with contronable flexibility. While more adaptable than a linear kernel, it can be sensitive to the chosen degree and may perform poorly when extrapolating beyond the training range.
- **Radial Basis Function (RBF) Kernel**: The most widely used kernel for non-linear problems, capable of generating highly flexible and smooth curves. It is versatile and effective for capturing complex relationships in the data.
:::

For the penguin regression task, we use the RBF kernel in the SVR model to capture potential non-linear relationships between flipper length and body mass that a simple linear model might not be able to detect.

```python
from sklearn.svm import SVR

svr_model = SVR(kernel='rbf', gamma=0.1, C=100.0, epsilon=1.0)
svr_model.fit(X_train_scaled, y_train_scaled)

y_pred_svr_scaled = svr_model.predict(X_test_scaled)
y_pred_svr = scaler_y.inverse_transform(y_pred_svr_scaled.reshape(-1, 1)).ravel()

rmse_svr = root_mean_squared_error(y_test_orig, y_pred_svr)
r2_value_svr = r2_score(y_test_orig, y_pred_svr)
print(f"Support Vector Regression RMSE: {rmse_svr:.2f}, R²: {r2_value_svr:.2f}")
```


:::{figure} ./images/6-regression-predictive-curve-linear-poly3-svr.png
:align: center
:width: 80%
:::


For the Penguins dataset, SVR can potentially outperform Linear Regression if the relationship between flipper length and body mass is non-linear, as it can flexibly adapt to complex patterns without requiring explicit polynomial features. However, in this regression task, SVR with a (non-linear) RBF kernel underperforms compared to the Linear Regression model. There are two main reasons for this:
- The relationship between flipper length and body mass is fundamentally linear or only mildly non-linear, so the flexibility of SVR is not necessary.
- The hyperparameters (``gamma``, `C`, `epsilon` in ``svr_model = SVR(kernel='rbf', gamma=0.1, C=100.0, epsilon=1.0)``) used for the SVR model may not be optimal. In this case, tuning the hyperparameters using techniques like grid search or cross-validation could improve performance.


::::{exercise} **Tuning hyperparameter**

In this exercise (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/6-ML-Regression.ipynb)), we will use **grid search** combined with **cross-validation** to find the optimal hyperparameters for the SVR model (code example is available in the Jupyter Notebook). We will:
- Compare RMSE and R² values to evaluate predictive performance.
- Plot predictive curves to visually assess how well the model fits the data.


:::{callout} **Grid search** and **Cross validation**
:class: dropdown

Grid search is a method used to find the best combination of hyperparameters for a ML model. It will search all possible combinations of the hyperparameter values you specify, trains the model for each combination, and evaluates it using a validation set. After testing all combinations, it picks the set of hyperparameters that gives the best performance.

Cross-validation is a method to check how well a model will perform on unseen data.
- Instead of just splitting the data once into training and testing sets, cross-validation splits the data into several parts (folds).
- The model is trained on some folds and tested on the remaining fold. This process is repeated so that every fold gets a turn as the test set.
- The performance scores from all folds are averaged, giving a more reliable estimate of how the model will do in real situations.
- Here is a one example: The dataset is split into 5 parts. The model trains on 4 parts and tests on 1 part. This is repeated 5 times, each time with a different test part.
:::
::::



### Decision Tree

In addition to instance-based models such as KNN and margin-based models like SVR, we can also apply tree-based methods for the regression task to predict a penguin's body mass from its flipper length. One of the most intuitive and interpretable approaches in this family is the **Decision Tree Regressor**.

A Decision Tree Regressor is a non-linear model that partitions the feature space (flipper length) into distinct regions based on feature thresholds and assigns a constant value (the average body mass) to each region. For the penguin regression task, the Decision Tree Regressor recursively splits the dataset into groups of penguins with similar flipper lengths. At each split, the model selects the threshold that minimizes the variance of body mass within the resulting groups. This recursive process continues until stopping criteria are met, such as reaching a maximum tree depth or a minimum number of samples per leaf.


Below is a code example demonstrating the Decision Tree Regressor (``max_depth = 3``) applied to the penguins regression task.

```python
from sklearn.tree import DecisionTreeRegressor

dt3_model = DecisionTreeRegressor(max_depth=3, random_state=123)
dt3_model.fit(X_train_scaled, y_train_scaled)

y_pred_dt3_scaled = dt3_model.predict(X_test_scaled)
y_pred_dt3 = scaler_y.inverse_transform(y_pred_dt3_scaled.reshape(-1, 1)).ravel()

rmse_dt3 = root_mean_squared_error(y_test_orig, y_pred_dt3)
r2_value_dt3 = r2_score(y_test_orig, y_pred_dt3)
print(f"Decision Tree (depth=3) RMSE: {rmse_dt3:.2f}, R²: {r2_value_dt3:.2f}")
```

Predictions for a new penguin are straightforward. Once the tree is built, the model follows the decision path down the tree according to the penguin’s feature values until it reaches a leaf node. The predicted value is then the mean (or median) body mass of the training samples in that leaf. For example, if a leaf node contains 15 penguins with an average body mass of 3850 grams, any new penguin whose features lead it to this leaf will be predicted to have a mass of 3850 g.

:::{figure} ./images/6-regression-predictive-curve-decision-tree-35.png
:align: center
:width: 80%
:::


When applying a Decision Tree Regressor to the penguin regression tesk, the tree depth plays a crucial role in shaping the fitted curve. With a relatively shallow tree, such as depth = 3, the model makes only a few splits on flipper length, resulting in broad, step-like regions where body mass is predicted as the average within each group. This provides a coarse approximation of the relationship, capturing the general trend but missing finer variations.

Increasing the tree depth to 5 allows for more splits, creating narrower regions and a fitted curve that follows the data more closely. While this improves flexibility and reduces bias, it also increases the risk of capturing noise in the training set, leading to overfitting. Comparing fitted curves at different depths illustrates **the classic trade-off in decision trees: shallow trees may underfit, while deeper trees may fit the training data too closely**.


:::{exercise} Random Forest and Gradient Boosting Regressions

We have discussed the limitations of Decision Tree algorithm, which can be mitigated using powerful ensemble methods such as Random Forest and Gradient Boosting. In this exercise (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/6-ML-Regression.ipynb)), we will:
- Apply Random Forest and Gradient Boosting Regressors to the penguin regression task using initial (arbitrary) hyperparameters.
- Optimize hyperparameters via grid search and cross-validation to improve predictive performance.
- Plot predictive curves to visually evaluate how well each model fits the data.
:::



### Multi-Layer Perceptron


For this penguins task, we will explore implementations using three popular frameworks: the user-friendly scikit-learn, the high-level deep learning library Keras, and the more granular, research-oriented PyTorch.

In Scikit-learn, the ``MLPRegressor`` class offers a convenient interface for training small- to medium-sized neural networks, requiring minimal configuration while still providing flexibility for most regression tasks.

```python
from sklearn.neural_network import MLPRegressor

mlp_model = MLPRegressor(hidden_layer_sizes=(32, 16, 8), activation='relu', 
                         solver='adam', max_iter=5000, random_state=123)
mlp_model.fit(X_train_scaled, y_train_scaled)

y_pred_mlp_scaled = mlp_model.predict(X_test_scaled)
y_pred_mlp = scaler_y.inverse_transform(y_pred_mlp_scaled.reshape(-1, 1)).ravel()

rmse_mlp = root_mean_squared_error(y_test_orig, y_pred_mlp)
r2_value_mlp = r2_score(y_test_orig, y_pred_mlp)
print(f"Multi-Layer Perceptron RMSE: {rmse_mlp:.2f}, R²: {r2_value_mlp:.2f}")
```

:::{figure} ./images/6-regression-predictive-curve-linear-mlp.png
:align: center
:width: 80%
:::

For greater control and scalability, frameworks like Keras (built on TensorFlow) and PyTorch allow us to design custom neural network architectures. We can specify the number of hidden layers, the number of neurons per layer, activation functions (*e.g.*, ReLU or tanh), and optimization algorithms (*e.g.*, stochastic gradient descent or Adam). These frameworks also offer tools for monitoring training, adjusting learning rates, and preventing overfitting through techniques such as regularization or dropout.

:::{exercise} DNNs using Keras (TensorFlow) and PyTorch

Code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/6-ML-Regression.ipynb):
- Construct DNNs using Keras and PyTorch.
- Apply DNNs to the penguin regression task using given hyperparameters.
- Optimize hyperparameters listed below to improve predictive performance
	- architecture hyperparameters
		- number of layers
		- number of neurons per layer
		- activation functions (*e.g.*, ``ReLU``, ``tanh``, ``sigmoid``)
	- training hyperparameters
		- optimizers (*e.g.*, SGD, Adam, RMSprop)
		- learning rate
		- batch size
		- number of epochs
	- regularization hyperparameters
		- dropout rate
		- early stopping parameters 
- Plot predictive curves for visualization
:::



## Comparison of Trained Models


:::{figure} ./images/6-rmse-r2-scores_alla.png
:align: center
:width:100%
:::


**Summary of regression models**
- Best performance: Polynomial Regression (degree=3) gave the lowest RMSE (407.47) and the highest R² (0.73), slightly outperforming Linear Regression.
- Strong performers: Linear Regression, Ridge, Lasso, and both Deep Neural Networks (Keras, PyTorch) all showed similar results (RMSE ≈ 412–415, R² ≈ 0.71–0.72).
- Moderate performers: Decision Trees (depth=3), Random Forest (depth=5), Gradient Boosting, and SVR (optimized) performed decently (R² ≈ 0.70–0.72) but not better than simpler models.
- Weaker performers: Plain SVR, Lasso, Gradient Boosting, and KNN trailed behind with higher RMSEs (>430) and lower R² values (≈0.68–0.69).



:::{seealso}
- [Hyperparameter optimization](https://en.wikipedia.org/wiki/Hyperparameter_optimization)
- [Grid search](https://drbeane.github.io/python_dsci/pages/grid_search.html)
- [Introduction to Cross-Validation in Machine Learning](https://thatdatatho.com/detailed-introduction-cross-validation-machine-learning/)
:::



:::{keypoints}
- We explored regression as a supervised learning task for predicting penguins body mass from their flipper length.
- Starting with simple models like Linear Regression, we gradually introduced more advanced approaches, including Polynomial Regression, Support Vector Regression, tree-based models, and neural networks.
- All models were evaluated using matrics including RMSE and R² scores, and visualized with predictive curves.
- Simple models (Linear and Polynomial Regression) performed as well as or better than complex models (SVR, trees, ensembles, neural network-based models).
- This indicates that the relationship between flipper length and body mass is mostly linear, with mild non-linear patterns.
:::
