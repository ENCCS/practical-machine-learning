# Supervised Learning (I): Classification



:::{objectives}
- Describe the basic concepts of classification tasks, including inputs (features), outputs (labels), and common algorithms.
- Preprocess data in the Penguins dataset by handling missing values, managing outliers, and encoding categorical features.
- Perform classification tasks using representative algorithms (*e.g.*, k-NN, Logistic Regression, Naive Bayes, Support Vector Machine, Decision Tree, Random Forest, Gradient Boosting, Multi-Layer Perceptron, and Neural Networks).
- Evaluate model performance with metrics such as accuracy, precision, recall, F1-score, and confusion matrices.
:::



:::{instructor-note}
- 40 min teaching/demonstration
- 40 min exercises
:::



## Classification


Classification is a supervised ML task in which a model predicts discrete class labels based on input features. It involves training the model on labeled data so that it can assign new and unseen data to predefined categories or classes by learning patterns from the training dataset.

In binary classification, the model predicts one of two classes, such as spam or not spam for emails. Multiclass classification extends this to multiple categories, like classifying images as cats, dogs, or birds.

Common algorithms for classification tasks include k-Nearest Neighbors (KNN), Logistic Regression, Naive Bayes, Support Vector Machine (SVM), Decision Trees, Random Forests, Gradient Boosting, and Neural Networks.

In this episode we will perform supervised classification to categorize penguins into three species --- Adelie, Chinstrap, and Gentoo --- based on their physical measurements (flipper length, body mass, *etc.*). We will build and train multiple classifier models, and then evaluate their performance using metrics such as accuracy, precision, recall, and F1 score.
By comparing the results, we aim to identify which model provides the most accurate and reliable classification for this task.



## Data Preparation


In the previous episode, [Episode 4: Data Preparation for Machine Learning](./04-data-preparation-for-ML.md), we discussed data preparation steps, including handling missing values, detecting outliers, and encoding categorical variables.

In this episode, we will revisit these steps, with particular emphasis on encoding categorical variables. For the classification task, we will treat the categorical variable ``species`` as the label (target variable) and use the remaining columns as features to predict the penguins species.
To achieve this, we transform the categorical features ``island`` and ``sex``, as well as the ``species`` label, into numerical format.

```python
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()

penguins = sns.load_dataset('penguins')
penguins_classification = penguins.dropna()

# encode `species` column with 0=Adelie, 1=Chinstrap, and 2=Gentoo
penguins_classification.loc[:, 'species'] = encoder.fit_transform(penguins_classification['species'])

# encode `island` column with 0=Biscoe, 1=Dream and 2=Torgersen
penguins_classification.loc[:, 'island'] = encoder.fit_transform(penguins_classification['island'])

# encode `sex` column with 0=Female, and 1=Male
penguins_classification.loc[:, 'sex'] = encoder.fit_transform(penguins_classification['sex'])
```


:::{discussion}
- why to use ``species``?
- why not to use the other categorical variables (``island`` and ``sex``)?
:::



## Data Processing

In this episode, data processing will focus on two essential steps: **data splitting** and **feature scaling**


### Data splitting

Data splitting involves two important substeps: splitting into features and labels, and splitting into training and testing sets.

The first substep is to split the dataset into features and labels. Features (also called predictors or independent variables) are the input values used to make predictions, while labels (or target variables) represent the output the model is trying to predict.

```python
X = penguins_classification.drop(['species'], axis=1)
y = penguins_classification['species'].astype('int')
```

The second substep is to divide the Penguins dataset into training and testing sets. The training set is used to fit and train the models, allowing it to learn patterns and relationships from the data, and the testing set, on the other hand, is reserved for evaluating the model’s performance on unseen data.

A common split is 80% for training and 20% for testing, which provides enough data for training while still retaining a meaningful set for testing.
This step is typically performed using the ``train_test_split`` function from ``sklearn.model_selection``, where setting a fixed ``random_state`` ensures reproducibility of the results.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

print(f"Number of examples for training is {len(X_train)} and test is {len(X_test)}")
```



### Feature scaling


Feature scaling is to standardize or normalize the range of independent variables (features) in a dataset.
In many datasets, features can have different units or scales. For example, in the Penguins dataset, body mass is measured in grams and can range in the thousands, while flipper length is measured in millimeters and typically ranges in the hundreds. These differences in scale can unintentionally bias ML algorithms, making features with larger values dominate the learning process, leading to biased and inaccurate models.

Scaling transforms these features to a common, limited range, such as [0, 1] or a distribution with a mean of 0 and a standard deviation of 1, without distorting the differences in the ranges of values or losing information.
This is particularly important for algorithms that rely on distance calculations, such as k-Nearest Neighbors (k-NN), Support Vector Machines (SVM), and clustering methods. Similarly, gradient-based optimization methods (used in neural networks and logistic regression) converge faster and more reliably when input features are scaled.
Without scaling, the algorithm might oscillate inefficiently or struggle to find the optimal solution. Furthermore, it helps ensure that regularization penalties are applied uniformly across all coefficients, preventing the model from unfairly penalizing features with smaller natural ranges.

Two of the most common methods for feature scaling are **Normalization** (Min-Max Scaling) and **Standardization** (Z-score Normalization).
- Normalization (Min-Max Scaling)
	- This technique rescales the features to a fixed range, typically [0, 1].
	- It is calculated by subtracting the minimum value of the feature and then dividing by the range (max - min), and its formula is 
		:::{math}
		X\_scaled = \frac{(X - X\_min)}{(X\_max - X\_min)}
		:::
	- This method is useful when the distribution is not Gaussian or when the algorithm requires input values bounded within a specific range (*e.g.*, neural networks often use activation functions that expect inputs in the [0,1] range).
- Standardization (Z-score Normalization)
	- This technique transforms the data to have a mean of 0 and a standard deviation of 1.
	- It is calculated by subtracting the mean value (μ) of the feature and then dividing by the standard deviation (σ), and its formula is
		:::{math}
		X\_scaled = \frac{X - \mu}{\sigma}.
		:::
	- Standardization is less affected by outliers than Min-Max scaling and is often the preferred choice for algorithms that assume data is centered (like SVM and PCA).

:::{figure} ./images/5-feature-scaling.png
:align: center
:width: 95%
:::

In practice, these transformations are easily applied using libraries like scikit-learn with the ``MinMaxScaler`` and ``StandardScaler`` classes, which efficiently learn the parameters (``mean``, ``min``, ``max``) from the training data and apply them consistently to avoid data leakage.

In this episode, we will apply feature standardization to both the training and testing sets. The implementation can be easily achieved using ``StandardScaler`` from ``sklearn.preprocessing``, as shown in the code below.
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```



## Training Model & Evaluating Model Performance


After preparing the Penguins dataset by handling missing values, encoding categorical variables, and splitting it into features/labels and training/testing sets, the next step is to apply classification algorithms.
In this episode, we will experiment with k-Nearest Neighbors (KNN), Naive Bayes, Decision Trees, Random Forests, and Neural Networks to predict penguins species based on their physical measurements. Each of these algorithms offers a distinct approach to pattern recognition and generalization. By applying them to the same prepared dataset, we can make a fair and meaningful comparison of their predictive performance.

The workflow for training and evaluating a classification model generally follows these steps:
- Choose a model class and import it, ``from sklearn.neighbors import XXX``.
- Set model hyperparameters by instantiating the class with desired values, ``xxx_model = XXX(<... hyperparameters ...>)``.
- Train the model on the preprocessed training data using the ``.fit()`` method, ``xxx_model.fit(X_train_scaled, y_train)``.
- Make predictions on the testing data with the ``.predict()`` method, ``y_pred_xxx = xxx_model.predict(X_test_scaled)``.
- Evaluate model performance using appropriate metrics, ``score_xxx = accuracy_score(y_test, y_pred_xxx)``.
- Visualize the results, for example by plotting a confusion matrix or other diagnostic charts to better understand model performance.



### k-Nearest Neighbors (KNN)


One intuitive and widely used method for classification is the k-Nearest Neighbors (KNN) algorithm. KNN is a non-parametric, instance-based approach that predicts a sample's label by considering the majority class of its *k* closest neighbors in the training set. Unlike many other algorithms, **KNN does not require a traditional training phase**; instead, **it stores the entire dataset and performs the necessary computations at prediction step**. This makes it a lazy learner --- simple to implement but potentially expensive during inference, especially with large datasets.

Below is an example illustrating how KNN determines the class of a new query point. Given a query point, KNN first calculates the distance between this point and all points in the training set. It then identifies the *k* closest points, and the class that appears most frequently among these neighbors is assigned as the predicted label for the query point. The choice of *k* plays a crucial role in performance: a small *k* can make the model overly sensitive to noise, while a large *k* may oversmooth the decision boundaries and obscure important local patterns.

:::{figure} ./images/5-knn-example.png
:align: center
:width: 90%
:::


Let’s create a KNN model. Here, we set ``k = 3``, meaning that the algorithm will consider the 3 nearest neighbors to determine the class of a data point. We then train the model on the training set using the ``.fit()`` method.
```python
from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=3)
knn_model.fit(X_train_scaled, y_train)
```

After fitting the model to the training dataset, we use the trained KNN model to predict the species on the testing set and evaluate its performance.

For classification tasks, metrics such as accuracy, precision, recall, and the F1-score provide a comprehensive assessment of model performance:
- **Accuracy** measures the proportion of correctly classified instances across all species (Adelie, Chinstrap, Gentoo). It provides an overall sense of how often the model is correct but can be misleading when the dataset is imbalanced.
- **Precision** quantifies the proportion of correct positive predictions for each species, while **recall** measures the proportion of actual positives that are correctly identified.
- **F1-score** is the harmonic mean of precision and recall, offering a balanced metric for each class. It is particularly useful when dealing with imbalanced class distributions, as it accounts for both false positives and false negatives.

:::{callout} Relations among different matrics
:class: dropdown

In classification tasks, model predictions can be compared against the true labels to assess performance. This comparison is often summarized using four key concepts: True Positive (TP), True Negative (TN), False Positive (FP), and False Negative (FN).

Suppose we focus on identifying Adelie penguins as the positive class.
- A True Positive occurs when the model correctly predicts a penguin as Adelie and it truly belongs to that species.
- A True Negative happens when the model correctly identifies a penguin as not Adelie (*i.e.*, Chinstrap or Gentoo).
- A False Positive arises when the model incorrectly predicts a penguin as Adelie when it is actually another species.
- A False Negative occurs when an Adelie penguin is mistakenly predicted as Chinstrap or Gentoo.

These four outcomes form the basis of performance metrics such as accuracy, precision, recall, and F1-score, which help evaluate how well the model distinguishes between species.
:::


```python
# predict on testing data
y_pred_knn = knn_model.predict(X_test_scaled)

# evaluate model performance
from sklearn.metrics import classification_report, accuracy_score

score_knn = accuracy_score(y_test, y_pred_knn)
print("Accuracy for k-Nearest Neighbors:", score_knn)
print("\nClassification Report:\n", classification_report(y_test, y_pred_knn))
```

In classification tasks, a **confusion matrix** is a powerful tool for evaluating model performance by comparing predicted labels with true labels. For a multiclass problem like the Penguins dataset, the confusion matrix is an **N x N** matrix, where N represents the number of target classes (here, **N=3** for the three penguins species).
Each cell *(i, j)* shows the number of instances where the true class was *i* and the model predicted class *j*. Diagonal elements correspond to correct predictions, while off-diagonal elements indicate misclassifications. This visualization provides an intuitive overview of how often the model predicts correctly and where it tends to make errors.

```python
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(conf_matrix, title, fig_name):
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='OrRd',
                xticklabels=["Adelie", "Chinstrap", "Gentoo"],
                yticklabels=['Adelie', 'Chinstrap', 'Gentoo'], cbar=True)
    
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(fig_name)


cm_knn = confusion_matrix(y_test, y_pred_knn)

plot_confusion_matrix(cm_knn, "Confusion Matrix using KNN algorithm", "5-confusion-matrix-knn.png")
```

:::{figure} ./images/5-confusion-matrix-knn.png
:align: center
:width: 75%

The first row: there are 28 Adelie penguins in the test data, and all these penguins are identified as Adelie (valid). The second row: there are 20 Chinstrap pengunis in the test data, with 2 identified as Adelie (invalid), and 18 identified as Chinstrap (valid). The third row: there are 19 Gentoo penguins in the test data, and all these penguins are identified as Gentoo (valid).
:::


:::{warning}
The choice of ``k`` can greatly affect the accuracy of KNN. Always try multiple ``k`` values and compare their performance. For the Penguins dataset, test different k values (*e.g.*, 3, 5, 7, 9, …) to find the optimal k that gives the best classification results (accuracy score).
:::



### Logistic Regression


**Logistic Regression** is a fundamental classification algorithm to predict categorical outcomes. Despite its name, logistic regression is not a regression algorithm but a classification method that predicts the **probability** of an instance belonging to a particular class.

For binary classification, it uses the logistic (**sigmoid**) function to map a linear combination of input features to a probability between 0 and 1, which is then thresholded (typically at 0.5) to assign a class.

For multiclass classification, logistic regression can be extended using approaches such as one-vs-rest (OvR) or softmax regression.
- In OvR, a separate binary classifier is trained for each species, treating that species as the positive class (blue area) and all other species as the negative class (red area).
- Softmax regression generalizes the logistic function to compute probabilities across all classes simultaneously, assigning each instance to the class with the highest predicted probability.

:::{figure} ./images/5-logistic-regression-example.png
:align: center
:width: 95%

(Upper left) the sigmoid function; (upper middle) the softmax regression process: three input features to the softmax regression model resulting in three output vectors where each contains the predicted probabilities for three possible classes; (upper right) a bar chart of softmax outputs in which each group of bars represents the predicted probability distribution over three classes; (lower subplots) three binary classifiers distinguish one class from the other two classes using the one-vs-rest approach.
:::

The process of creating a Logistic Regression model and fitting it to the training data is very similar to the approach used for the KNN model described earlier, with the main difference being the choice of classifier. A code example and the resulting confusion matrix plot are provided below.

```python
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(random_state = 123)
lr_model.fit(X_train_scaled, y_train)

y_pred_lr = lr_model.predict(X_test_scaled)

score_lr = accuracy_score(y_test, y_pred_lr)
print("Accuracy for Logistic Regression:", score_lr )
print("\nClassification Report:\n", classification_report(y_test, y_pred_lr))

cm_lr = confusion_matrix(y_test, y_pred_lr)

plot_confusion_matrix(cm_lr, "Confusion Matrix using Logistic Regression algorithm", "5-confusion-matrix-lr.png")
```

:::{figure} ./images/5-confusion-matrix-lr.png
:align: center
:width: 75%
:::



### Naive Bayes


The **Naive Bayes** algorithm is a simple yet powerful probabilistic classifier based on [Bayes' Theorem](https://en.wikipedia.org/wiki/Bayes%27_theorem). It assumes that all features are and equally important --- a condition that often does not hold in practice, which can introduce some bias. However, this independence assumption greatly simplifies computations by allowing conditional probabilities to be expressed as the product of individual feature probabilities. Given an input instance, the algorithm calculates the posterior probability for each class and assigns the instance to the class with the highest probability.

Logistic Regression and Naive Bayes are both popular algorithms for classification tasks, but they differ significantly in their approach, assumptions, and underlying mechanics. Below is an example comparing Logistic Regression and Naive Bayes decision boundaries on a synthetic dataset with two features. The visualization highlights their fundamental differences: **Logistic Regression learns a linear decision boundary directly, whereas Naive Bayes models feature distributions for each class under the independence assumption**.


:::{callout} **Logistic Regression** *vs.* **Naive Bayes**
:class: dropdown

- Logistic Regression is a **discriminative** model that directly estimates the probability of a data point belonging to a particular class by fitting a linear combination of features. In the context of the Penguins dataset, Logistic Regression uses features such as bill length and flipper length to compute a weighted sum, which is then transformed into probabilities for penguins species. The model assumes a linear relationship between the features and the log-odds of the classes and optimizes parameters using maximum likelihood estimation. This makes Logistic Regression sensitive to feature scaling and correlations. It is generally robust to noise and can tolerate moderately correlated features, but it may struggle with highly non-linear relationships unless additional feature engineering is applied.
- Naive Bayes, by contrast, is a **generative** model that applies Bayes’ theorem to estimate the probability of a class given the input features, assuming conditional independence between features. For the Penguins dataset, it estimates the likelihood of features (*e.g.*, bill depth) for each species and combines these with prior probabilities to predict the most likely species. The "naive" independence assumption often does not hold in practice (*e.g.*, bill length and depth may be correlated), but it simplifies computation and allows Naive Bayes to be highly efficient, especially for high-dimensional data. It is less sensitive to irrelevant features and does not require feature scaling. However, it can underperform when feature dependencies are strong or when the data distribution deviates from the model’s assumptions (*e.g.*, Gaussian for continuous features in Gaussian Naive Bayes). Zero probabilities must be carefully handled, typically via smoothing techniques.
:::

:::{figure} ./images/5-naive-bayes-example.png
:align: center
:width: 95%
:::


To apply Naive Bayes, we use ``GaussianNB`` from ``sklearn.naive_bayes``, which assumes that the features follow a Gaussian (normal) distribution --- making it suitable for continuous numerical data such as bill length and body mass.
- Because Naive Bayes relies on probabilities, **feature scaling is not required**; however, **handling missing values and encoding categorical variables numerically remains necessary**.
- While Naive Bayes may not always match the performance of more complex models like Random Forests, it offers fast training, low memory requirements, and reliable performance for simpler classification tasks.

```python
from sklearn.naive_bayes import GaussianNB

nb_model = GaussianNB()
nb_model.fit(X_train_scaled, y_train)

y_pred_nb = nb_model.predict(X_test_scaled)

score_nb = accuracy_score(y_test, y_pred_nb)
print("Accuracy for Naive Bayes:", score_nb)
print("\nClassification Report:\n", classification_report(y_test, y_pred_nb))

cm_nb = confusion_matrix(y_test, y_pred_nb)
plot_confusion_matrix(cm_nb, "Confusion Matrix using Naive Bayes algorithm", "4-confusion-matrix-nb.png")
```

:::{figure} ./images/5-confusion-matrix-nb.png
:align: center
:width: 75%
:::



### Support Vector Machine (SVM)


Previously, we presented an example using a Logistic Regression classifier, which produces a linear decision boundary to separate two classes based on their features. It works by fitting this linear boundary using the logistic function, making it particularly effective when the data is linearly separable. A notable characteristic of Logistic Regression is that the decision boundary typically lies in the region where the predicted probabilities of the two classes are closest --- essentially where the model is most uncertain.

However, when there is a large gap between two well-separated classes --- as can occur when distinguishing cats from dogs based on weight and size --- Logistic Regression faces an inherent limitation: an infinite number of possible solutions. The algorithm has no built-in mechanism to select a single "optimal" boundary when multiple valid linear separators exist within the wide margin between classes. As a result, it may place the decision boundary somewhere in that gap, creating a broad, undefined region with little or no supporting data. While this may not affect accuracy on clearly separated data, it can reduce the model’s robustness when new or noisy data points appear near that boundary.

Below is another example of separating cats from dogs based on ear length and weight. In addition to the linear decision boundary produced by the Logistic Regression classifier, we can identify three other linear boundaries that also achieve good separation between the two classes. The question then arises: which boundary is truly better, and how can we evaluate their performance on unseen data?

:::{figure} ./images/5-svm-example-large-gap.png
:align: center
:width: 95%
:::


To better handle such situations, we can turn to the **Support Vector Machine (SVM)** algorithm. Unlike Logistic Regression, SVM focuses on maximizing the margin --- the distance between the decision boundary and the closest data points from each class, known as support vectors (as illustrated in the figure below). When a large gap exists between two classes, SVM takes advantage of this space by positioning the boundary near the center of the gap while maintaining the maximum margin. This results in a more stable and robust classifier, especially when the classes are well-separated.

Unlike Logistic Regression, which considers all data points to estimate probabilities, SVM relies primarily on the most critical examples --- those closest to the decision boundary --- making it less sensitive to outliers and more precise in defining class separations.

:::{figure} images/5-svm-example-with-max-margin-separation.png
:align: center
:width: 95%

The SVM classification boundary for distinguishing cats and dogs based on ear length and weight. The solid black line represents the maximum margin hyperplane (decision boundary), while the dashed green lines indicate the positive and negative hyperplanes that define the margin. The black circles highlight the support vectors --- the critical data points that determine the width of the margin.
:::


To apply SVM, we use ``SVC`` (Support Vector Classification) from ``sklearn.svm``. By default, it assumes a nonlinear relationship between features, modeled using the ``rbf`` (Radial Basis Function) kernel. This kernel enables the model to learn complex decision boundaries by implicitly mapping input features into a higher-dimensional space.

:::{note}
You can also experiment with other kernels, such as ``linear``, ``poly``, or ``sigmoid``, to explore different types of decision boundaries.
:::

By adjusting hyperparameters such as ``C`` (regularization strength) and ``gamma`` (kernel coefficient), we can control the trade-off between margin width and classification accuracy. Below is a code example demonstrating how to apply ``SVC`` with the ``rbf`` kernel to classify penguins.

```python
from sklearn.svm import SVC

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=123)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)

score_svm = accuracy_score(y_test, y_pred_svm)
print("Accuracy for Support Vector Machine:", score_svm)
print("\nClassification Report:\n", classification_report(y_test, y_pred_svm))

cm_svm = confusion_matrix(y_test, y_pred_svm)

plot_confusion_matrix(cm_svm, "Confusion Matrix using Support Vector Machine algorithm", "5-confusion-matrix-svm.png")
```

:::{figure} ./images/5-confusion-matrix-svm.png
:align: center
:width: 75%
:::



### Decision Tree


The **Decision Tree** algorithm is a versatile and highly interpretable method for classification tasks. Its core idea is to recursively split the dataset into smaller subsets based on feature thresholds, creating a tree-like structure of decisions that maximizes the separation of target classes.

For example, a decision tree can be used to classify cats and dogs based on two or three features, illustrating how the algorithm partitions the feature space to distinguish between classes.

:::{figure} ./images/5-decision-tree-example.png
:align: center
:width: 95%

(Upper): Decision boundary separating cats and dogs based on two features (ear length and weight), along with the corresponding decision tree structure.
(lower): Decision boundaries separating cats and dogs based on three features (ear length, weight, and tail length), and the corresponding decision tree structure.
:::


Below is a code example demonstrating the Decision Tree classifier applied to the penguins classification task.

```python
from sklearn.tree import DecisionTreeClassifier

dt_model = DecisionTreeClassifier(max_depth=3, random_state = 123)
dt_model.fit(X_train_scaled, y_train)

y_pred_dt = dt_model.predict(X_test_scaled)

score_dt = accuracy_score(y_test, y_pred_dt)
print("Accuracy for Decision Tree:", score_dt )
print("\nClassification Report:\n", classification_report(y_test, y_pred_dt))

cm_dt = confusion_matrix(y_test, y_pred_dt)
plot_confusion_matrix(cm_dt, "Confusion Matrix using Decision Tree algorithm", "5-confusion-matrix-dt.png")
```

:::{figure} ./images/5-confusion-matrix-dt.png
:align: center
:width: 75%
:::


We visualize the Decision Tree structure to better understand how penguins are classified based on their physical characteristics.

```python
from sklearn.tree import plot_tree

plt.figure(figsize=(16, 6))
plot_tree(dt_model, feature_names=X.columns, filled=True, rounded=True, fontsize=10)

plt.title("Decision Tree Structure for Penguins Species Classification", fontsize=16)

plt.tight_layout()
plt.show()
```

:::{figure} ./images/5-decision-tree-structure.png
:align: center
:width: 100%
:::



### (Optional) Random Forest


While Decision Trees are easy to interpret and visualize, they have some notable drawbacks. One primary issue is their tendency to overfit the training data, particularly when the tree is allowed to grow deep without constraints such as maximum depth or minimum samples per split. Overfitting causes the model to capture noise in the training data, which can lead to poor generalization on unseen data --- for example, misclassifying a Gentoo penguin as a Chinstrap due to overly specific splits. Additionally, decision trees are sensitive to small variations in the data; even slight changes, such as a few noisy measurements, can result in a significantly different tree structure, reducing the model’s stability and reliability.

To address these limitations, we can use an ensemble learning technique called **Random Forest**. A Random Forest builds on the concept of decision trees by creating a large collection of them, each trained on a randomly selected subset of the data and features. By aggregating the predictions of multiple trees --- typically through majority voting for classification --- Random Forest reduces overfitting, improves generalization, and mitigates the inherent instability in individual decision trees.

:::{note}
**Ensemble learning** is a ML approach that combines multiple individual models (often called base learners) to create a stronger, more accurate, and more robust overall model. The idea is that by aggregating the predictions of several models, the ensemble can reduce errors, improve generalization, and mitigate weaknesses of individual models. There are two main types of ensemble learning techniques:
- **Bagging** (**Bootstrap Aggregating**): Multiple models are trained independently on random subsets of the data, and their predictions are averaged (for regression) or voted on (for classification). Random Forest is a classic example of bagging applied to decision trees.
- **Boosting**: Models are trained sequentially, with each new model focusing on the errors made by previous models. Examples include AdaBoost, Gradient Boosting, and XGBoost.
:::


The figure below illustrates how a Random Forest improves upon a single Decision Tree when classifying cats and dogs based on synthetic measurements of ear length and weight.

:::{figure} ./images/5-random-forest-example.png
:align: center
:width: 100%
   
Top row shows the classification boundaries for both models. On the left, a single Decision Tree creates rigid, rectangular decision regions that precisely follow axis-aligned splits in the training data. While this achieves a good separation of the training samples, the jagged boundaries suggest potential overfitting to noise. In contrast, the Random Forest (right) produces smoother, more nuanced decision boundaries through majority voting across 100 trees. The blended purple transition zones represent areas where individual trees disagree, demonstrating how the ensemble averages out erratic predictions from any single tree. Bottom row reveals why Random Forests are more robust by examining three constituent trees. Tree #1 prioritizes ear length for its initial split, Tree #2 begins with weight, and Tree #3 uses a completely different weight threshold.
:::


Below is a code example demonstrating the application of the Random Forest classifier to the penguins classification task.
```python
from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=100, random_state=123)
rf_model.fit(X_train_scaled, y_train)

y_pred_rf = rf_model.predict(X_test_scaled)

score_rf = accuracy_score(y_test, y_pred_rf)
print("Accuracy for Random Forest:", score_rf )
print("\nClassification Report:\n", classification_report(y_test, y_pred_rf))

cm_rf = confusion_matrix(y_test, y_pred_rf)

plot_confusion_matrix(cm_rf, "Confusion Matrix using Random Forest algorithm", "5-confusion-matrix-rf.png")
```

:::{figure} ./images/5-confusion-matrix-rf.png
:align: center
:width: 75%
:::


In addition to the confusion matrix, feature importance in a Random Forest (and also in a Decision Tree) model provides valuable insight into which input features contribute most to the model’s predictions. Random Forest calculates feature importance by measuring how much each feature reduces impurity --- such as Gini impurity or entropy --- when used to split the data across all trees in the forest. Features that produce greater reductions in impurity are considered more important. These importance scores are then normalized to provide a relative ranking, helping to identify which features most strongly influence the model’s predictions. This information is particularly useful for interpreting model behavior, selecting meaningful features, and understanding the underlying structure of the data.

:::{note}
In Decision Tree and Random Forest, impurity measures how "mixed" the classes are in a given node. A pure node contains only instances of a single class, while an impure node contains a mixture of classes. Impurity metrics help the tree decide which feature and threshold to use when splitting the data to create nodes that are as pure as possible.

Gini impurity and entropy are metrics used to measure impurity of a dataset or a node.

During training, the algorithm evaluates all possible splits for a feature. It chooses the split that maximizes purity, *i.e.*, **minimizes Gini impurity** or maximizes information gain (**reduction in entropy**).
:::


The greater the total reduction in impurity attributed to a feature, the more important it is considered. These importance scores are then normalized to provide a relative ranking, helping identify which features have the most influence on predicting the output class. This information is particularly useful for interpreting model behavior, selecting meaningful features, and understanding the underlying structure of the data.

Below is a code example showing how to plot feature importance using a Random Forest model to classify penguins into three categories.
```python
importances = rf_model.feature_importances_
features = X.columns

plt.figure(figsize=(9, 6))
plt.barh(features, importances, color="tab:orange", alpha=0.75)
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.show()
```

:::{figure} ./images/5-random-forest-feature-importrance.png
:align: center
:width: 80%

Illustration of feature importance for penguin classification. Longer bars indicate features with greater influence on the model’s decisions, showing that the Random Forest relies more heavily on these measurements to identify species.
:::



### (Optional) Gradient Boosting


We have trained the model using a Decision Tree classifier, providing an intuitive starting point for classifying penguin species based on physical measurements. However, this classifier is sensitive to small fluctuations in the dataset, which can often lead to overfitting, especially when the tree grows deep.

To address the limitations of a single decision tree, we turned to Random Forest, an ensemble method that builds multiple decision trees on different random subsets of the data and features. By averaging the predictions of all trees or taking a majority vote in classification, Random Forest reduces overfitting and improves generalization. This approach balances model complexity with predictive performance and provides a reliable estimate of feature importance, helping identify which physical attributes are most influential in distinguishing penguin species.

While Random Forest provides robustness and improved accuracy over individual trees, we can further enhance performance using **Gradient Boosting**.
- Like Random Forest, Gradient Boosting is an ensemble learning technique, but it builds a strong classifier by combining many weak learners (typically shallow decision trees) in a sequential manner.
- Unlike Random Forest, which grows multiple trees independently and in parallel using random subsets of the training data, Gradient Boosting constructs trees one at a time, with each new tree trained to correct the errors of its predecessors.

:::{figure} ./images/5-random-forest-vs-gradient-boosting.png
:align: center
:width: 80%
   
Iillustration of the [Random Forest](https://medium.com/@mrmaster907/introduction-random-forest-classification-by-example-6983d95c7b91) and [Gradient Boosting](https://bmcmedinformdecismak.biomedcentral.com/articles/10.1186/s12911-021-01701-9) algorithms.
:::


In this code example below, we apply Gradient Boosting algorithm to classify penguin species. We use ``GradientBoostingClassifier`` from scikit-learn due to its simplicity and strong baseline performance.

```python
from sklearn.ensemble import GradientBoostingClassifier

gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=123)
gb_model.fit(X_train_scaled, y_train)

y_pred_gb = gb_model.predict(X_test_scaled)

score_gb = accuracy_score(y_test, y_pred_gb)
print("Accuracy for Gradient Boosting:", score_gb)
print("\nClassification Report:\n", classification_report(y_test, y_pred_gb))

cm_gb = confusion_matrix(y_test, y_pred_gb)
plot_confusion_matrix(cm_gb, "Confusion Matrix using Gradient Boosting algorithm", "5-confusion-matrix-gb.png")
```

:::{figure} ./images/5-confusion-matrix-gb.png
:align: center
:width: 75%
:::

:::{note}

This progression --- from the simplicity of a single Decision Tree, to the robustness of Random Forest, and finally to the precision of Gradient Boosting --- mirrors the evolution of **tree-based methods** in modern ML. While Random Forest remains excellent for baseline performance, Gradient Boosting often achieves state-of-the-art results on structured data, such as ecological measurements, provided the learning rate and tree depth are carefully tuned.
:::



### Multi-Layer Perceptron


A **Multilayer Perceptron** (MLP) is a type of artificial neural network consisting of multiple layers of interconnected perceptrons (or neurons) designed to mimic certain aspects of human brain function. Each neuron (illustrated in the figure below) has the following characteristics:
- Input: one or more inputs (`x_1`, `x_2`, ...), *e.g.*, features from the input data expressed as floating-point numbers.
- Operations: Typically, each neuron conducts three main operations:
	- Compute the weighted sum of the inputs where (`w_1`, `w_2`, ...) are the corresponding weights.
	- Add a bias term to the weighted sum.
	- Apply an activation function to the result.
- Output: The neuron produces a single output value.

:::{figure} ./images/5-neuron-activation-function.png
:align: center
:width: 80%
:::

A common equation for the output of a neuron is
:::{math}
output = Activation(\sum_i (x_i * w_i) + bias).
:::

An **activation function** is a mathematical transformation that converts the weighted sum of a neuron’s inputs into its output signal. By introducing non-linearity into the network, activation functions enable neural networks to learn complex patterns and make sophisticated decisions based on the weighted inputs.

Below are some commonly used activation functions in neural networks and DL models. Each plays a crucial role in introducing non-linearities, allowing the network to capture intricate patterns and relationships in data.
- **Sigmoid**: With its characteristic S-shaped curve, the sigmoid function maps inputs to a smooth 0-1 range, making it historically popular for binary classification tasks.
- **Hyperbolic tangent** (tanh): Similar to sigmoid but ranging from -1 to 1, tanh often provides stronger gradients during training.
- **Rectified Linear Unit** (ReLU): Outputs zero for negative inputs and the identity for positive inputs. ReLU has become the default choice for many architectures due to its computational efficiency and its ability to mitigate the vanishing gradient problem.
- **Linear**: This identity function serves as a reference, showing network behavior without any non-linear transformation.

:::{figure} ./images/5-activation-function.png
:align: center
:width: 80%
:::


A single neuron (perceptron) can learn simple patterns but is limited in modeling complex relationships. By combining multiple neurons into layers and connecting them into a network, we create a powerful computational framework capable of approximating highly non-linear functions. In a Multilayer Perceptron (MLP), neurons are organized into an input layer, one or more hidden layers, and an output layer.

The image below illustrates a three-layer perceptron network with 3, 4, and 2 neurons in the input, hidden, and output layers, respectively.
- The input layer receives raw data, such as pixel values or measurements, and passes it to the hidden layer.
- The hidden layer contains multiple neurons that process the information and progressively extract higher-level features. Each neuron in the hidden layer is fully connected to neurons in adjacent layers, forming a dense network of weighted connections.
- The output layer produces the network’s predictions, whether it's a classification, regression output, or some other task.

:::{figure} ./images/5-mlp-network.png
:align: center
:width: 80%
:::


In the penguin classification task, we build a three-layer perceptron using scikit-learn’s ``MLPClassifier`` from ``sklearn.neural_network``.

```python
from sklearn.neural_network import MLPClassifier

mlp_model = MLPClassifier(hidden_layer_sizes=(16), activation='relu', solver='adam',
                   alpha=0, batch_size=8, learning_rate='constant',
                   learning_rate_init=0.001, max_iter=1000,
                   random_state=123, n_iter_no_change=10)
mlp_model.fit(X_train_scaled, y_train)
```

The model is configured with:
- an input layer matching the number of features (6 per penguin),
- a hidden layer (*e.g.*, 16 neurons) to capture non-linear relationships, and
- an output layer with three nodes (one per penguin class), using ``relu`` activation for the hidden layer.


The hyperparameters used to construct this MLP are listed below:
- ``adam``, the optimization algorithm used to update weight parameters.
- ``alpha``, the L2 regularization term (penalty). Setting this to 0 disables regularization, meaning the model won’t penalize large weights. This may cause overfitting if the dataset is small or noisy.
- ``batch_size``, the number of samples per mini-batch during training. Smaller batches lead to more frequent updates (finer learning) but can increase noise and training time.
- ``learning_rate``, specifies the learning rate schedule. "constant" means that the learning rate keeps fixed throughout training. Other options like "invscaling" or "adaptive" would adjust the learning rate during training.
- ``learning_rate_init=0.001``, the initial learning rate (fixed here). A smaller value means slower learning, which may require more iterations but offers more stability. 
- ``max_iter``, the maximum number of training iterations (epochs).
- ``random_state=123``, controls the random number generation for weight initialization and data shuffling, ensuring reproducible results.
- ``n_iter_no_change=10``, if the validation score does not improve for 10 consecutive iterations, training will stop early. This is a form of early stopping to prevent overfitting or unnecessary computation.


After training the model, we evaluate its accuracy on the testing set and visualize the results by computing and plotting the confusion matrix.
```python
y_pred_mlp = mlp_model.predict(X_test_scaled)

score_mlp = accuracy_score(y_test, y_pred_mlp)
print("Accuracy for Neural Network:", score_mlp)
print("\nClassification Report:\n", classification_report(y_test, y_pred_mlp))

cm_mlp = confusion_matrix(y_test, y_pred_mlp)
plot_confusion_matrix(cm_mlp, "Confusion Matrix using Multi-Layer Perceptron algorithm", "5-confusion-matrix-mlp.png")
```


:::{figure} ./images/5-confusion-matrix-mlp.png
:align: center
:width: 75%
:::



### (Optional) Deep Neural Networks

MLP is a foundational neural network architecture, consisting of an input layer, one or more hidden layers, and an output layer. While MLP excels at learning complex patterns from tabular data, its shallow depth (typically 1-2 hidden layers) limits its ability to handle very high-dimensional or abstract data such as raw images, audio, or text.

To overcome these limitations, Deep Neural Network (DNN) extends the MLP framework by adding multiple hidden layers. These additional layers allow the model to learn highly abstract features through deep hierarchical representations: early layers might capture basic features (like edges or shapes), while deeper layers recognize complex objects or semantic patterns. This depth enables DNN to outperform traditional MLP in complex tasks requiring high-level feature extraction, such as computer vision and natural language processing.

:::{callout} DNN architectures

DNNs have specialized architectures designed to handle different types of data (*e.g.*, spatial, temporal, and sequential data) and tasks more effectively.
- A standard feedforward deep neural network consists of stacked fully connected layers
- **Convolutional neural networks** (CNNs) are particularly well-suited for image data. They use convolutional layers to automatically extract local features like edges, textures, and shapes, significantly reducing the number of parameters and improving generalization on visual tasks.
- **Recurrent neural network** (RNN) is designed for sequential data such as time series, speech, or natural language. RNNs include loops that allow information to persist across time steps, enabling the model to learn dependencies over sequences. More advanced versions, like Long Short-Term Memory (LSTM) networks and Gated Recurrent Units (GRUs), address the limitations of basic RNNs by managing long-term dependencies more effectively.
- In addition to CNNs and RNNs, the **Transformer** architecture has emerged as the state-of-the-art in many language and vision tasks. Transformers rely entirely on attention mechanisms rather than recurrence or convolutions, enabling them to model global relationships in data more efficiently. This flexibility has made them the foundation of powerful models like BERT, GPT, and Vision Transformers (ViTs). These specialized DL architectures illustrate how tailoring the network design to the structure of the data can lead to significant performance gains and more efficient learning.
:::


Here, we use the Keras API to construct a small DNN and apply it to the penguin classification task, demonstrating how even a compact architecture can effectively distinguish between penguin species (Adelie, Chinstrap, and Gentoo).


In this example, we exclude the categorical features ``island`` and ``sex`` from both the training and testing datasets. The target label ``species`` is then encoded using the ``pd.get_dummies()`` function in Pandas. Afterward, we split the data into training and testing sets and standardize the feature values to ensure consistent scaling during model training.

```python
from tensorflow import keras

X = penguins_classification.drop(['species','island', 'sex'], axis=1)
y = penguins_classification['species'].astype('int')
y = pd.get_dummies(penguins_classification['species']).astype(np.int8)
y.columns = ['Adelie', 'Chinstrap', 'Gentoo']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
print(f"Number of examples for training is {len(X_train)} and test is {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

When building a DNN with Keras, there are two common approaches: using the ``Sequential()`` API step by step, or defining all layers at once within the ``Sequential()`` constructor. Here, we adopt the first approach, whereas the second approach is used in the Jupyter notebook to construct the same DNN.
- We start by creating an empty model with ``keras.Sequential()``, which initializes a linear container for stacking sequential layers. 
- Next, we define each layer separately using the ``Dense`` class, specifying the number of neurons and activation function for each layer.
- Finally, we add all the layers using ``keras.Model()`` to the sequential container, resulting in a trainable model.

```python
from tensorflow.keras.layers import Dense, Dropout

dnn_model = Sequential()

input_layer = keras.Input(shape=(X_train_scaled.shape[1],)) # 4 input features

hidden_layer1 = Dense(32, activation="relu")(input_layer)
hidden_layer1 = Dropout(0.2)(hidden_layer1)

hidden_layer2 = Dense(16, activation="relu")(hidden_layer1)
#hidden_layer2 = Dropout(0.0)(hidden_layer2)

hidden_layer3 = Dense(8, activation="relu")(hidden_layer2)

output_layer = Dense(3, activation="softmax")(hidden_layer3) # 3 classes

dnn_model = keras.Model(inputs=input_layer, outputs=output_layer)
```


The ``keras.layers.Dropout()`` is a regularization technique in Keras used to reduce overfitting by randomly setting a fraction of input units to zero during training. For example, ``Dropout(0.2)`` means that 20% of the outputs of a specific layer will be randomly set to zero in each training step.

:::{figure} ./images/5-dnn-network-dropout.png
:align: center
:width: 80%
:::


We can use ``dnn_model.summary()`` to print a concise summary of a DNN’s architecture. It provides provides an overview of the model’s layers, their output shapes, and the number of trainable parameters, making it easier to understand and debug the network.

:::{figure} ./images/5-dnn-summary.png
:align: center
:width: 80%
:::


Now that we have designed a DNN that, in theory, should be capable of classifying penguins, we need to specify two critical components before training: (1) a loss function to quantify prediction errors, and (2) an optimizer to adjust the model’s weights during training.
- **Loss function**: For multi-class classification, we select categorical cross-entropy, which penalizes incorrect probabilistic predictions. In Keras, this is implemented via the ``keras.losses.CategoricalCrossentropy`` class. This loss function works naturally with the ``softmax`` activation function we applied in the output layer. For a full list of available loss functions in Keras, see the [documentation](https://www.tensorflow.org/api_docs/python/tf/keras/losses>).
- **Optimizer**: The optimizer determines how efficiently the model converges during training. Keras provides many options, each with its advantages, but here we use the widely adopted ``Adam`` (adaptive moment estimation) optimizer. Adam has several parameters, and the default values generally perform well, so we will use it with its defaults.


We use ``model.compile()`` to combine the chosen loss function and optimier before starting training.
```python
from keras.optimizers import Adam

dnn_model.compile(optimizer='adam', loss=keras.losses.CategoricalCrossentropy())
```

Now we are ready to train the DNN model. Here, we vary only the number of ``epochs``. One training epoch means that every sample in the training data has been shown to the neural network once and used to update its parameters. During training, we set ``batch_size=16`` to balance memory efficiency with gradient stability, and ``verbose=1`` to display a progress bar showing the loss and metrics for each epoch in real time.

```python
history = dnn_model.fit(X_train_scaled, y_train, batch_size=16, epochs=100, verbose=1)
```

The ``.fit()`` method returns a history object, which contains a history attribute holding the training loss and other metrics for each epoch. Plotting the training loss can provide valuable insight into how learning progresses. For example, we can use Seaborn to plot the training loss with epochs ``sns.lineplot(x=history.epoch, y=history.history['loss'], c="tab:orange", label='Training Loss')``.

:::{figure} ./images/5-dnn-loss.png
:align: center
:width: 80%
::: 


Finally, we evaluate the model’s performance on the testing set by computing its accuracy and visualizing the results with a confusion matrix.
```python
# predict class probabilities
y_pred_dnn_probs = dnn_model.predict(X_test_scaled)

# convert probabilities to class labels
y_pred_dnn = np.argmax(y_pred_dnn_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

score_dnn = accuracy_score(y_true, y_pred_dnn)
print("Accuracy for Deep Neutron Network:", score_dnn)
print("\nClassification Report:\n", classification_report(y_true, y_pred_dnn))

cm_dnn = confusion_matrix(y_true, y_pred_dnn)
plot_confusion_matrix(cm_dnn, "Confusion Matrix using DNN algorithm", "5-confusion-matrix-dnn.png")
```

:::{figure} ./images/5-confusion-matrix-dnn.png
:align: center
:width: 75%
:::



## Comparison of Trained Models

To evaluate the performance of different algorithms in classifying penguin species, we compare their accuracy scores and confusion matrices. The algorithms we the adopted in the previous sections include:
- Instance-based: k-Nearest Neighbors (KNN).
- Probability-based: Logistic Regression, and Naive Bayes.
- Hyperplane-based: Support Vector Machine (SVM).
- Tree-based methods: Decision Tree, Random Forest, and Gradient Boosting.
- Network-based models: Multi-Layer Perceptron (MLP) and Deep Neural Networks (DNN).

Each model was trained on the same training set and evaluated on a common testing set, with consistent preprocessing applied across all methods.


Performance under current training settings:
- MLP achieved the highest accuracy, demonstrating its effectiveness in capturing complex patterns and feature interactions in the Penguins dataset.
- Naive Bayes showed slightly lower accuracy, likely due to its strong independence assumption between features, which does not fully hold in this dataset.
- The other algorithms provided moderate performance.

:::{figure} ./images/5-scores-for-all-models.png
:align: center
:width: 80%
:::


The confusion matrices provided deeper insight into class-level prediction performance:
- MLP demonstrated well-balanced performance across all three penguin species.
- Naive Bayes, in contrast, confused Adelie and Chinstrap penguins, likely due to overlapping feature distributions between these species.
- other algorithms had a limited number of misclassifications, primarily between Adelie and Chinstrap.

:::{figure} ./images/5-compare-confusion-matrices.png
:align: center
:width: 100%
:::


:::{seealso}
- [Introduction to Deep Learning](https://enccs.github.io/deep-learning-intro/)
:::


:::{keypoints}
- Provided a fundamental introducton to classification tasks, covering basic concepts.
- Demonstrated essential steps for data preparation and processing using the Penguins dataset.
- Applied a range of classification algorithms --- instance-based, probability-based, margin-based, tree-based, and neural network-based --- to classify penguin species.
- Evaluated and compared model performance using metrics such as accuracy scores and confusion matrices.
:::
