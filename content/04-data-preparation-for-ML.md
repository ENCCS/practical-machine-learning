# Data Preparation for Machine Learning



:::{objectives}
- Provide an overview of data preparation.
- Load the Penguins dataset.
- Use pandas and seaborn to analyze and visualize the data.
- Identify and manage missing values and outliers in the dataset.
- Encode categorical variables into numerical values suitable for machine learning models.
:::



:::{instructor-note}
- 30 min teaching
- 20 min exercising
:::



In [Episode 2: Fundamentals of Machine Learning](./02-fundamentals-of-ML.md), it is clearly shown that data preparation and processing often consume a significant portion of the ML workflow --- often more time than the actual model training, evaluation, and optimization. 
Cleaning, transforming, and structuring raw data into a usable format ensures that algorithms can efficiently extract valuable insights.
Additionally, the choice of data formats, such as CSV for simplicity or HDF5 for large-scale datasets, can significantly impact data storage, accessibility, and computational efficiency during both model training and deployment.

In this episode, we will provide an overview of data preparation and introduce available public datasets. Using the Penguins dataset as an example, we will offer demonstrations and hands-on exercises to develop a comprehensive understanding of data preparation including handling missing values and outliers, encoding categorical variables, and other essential preprocessing techniques for ML workflows.



## What is Data Preparation


Data preparation refers to the process of cleaning, structuring, and transforming raw data into a structured, high-quality format ready for statistical analysis and ML. It’s one of the most critical steps in the ML workflow because high-quality data leads to better model performance. Key procedures include:
- collecting data from multiple sources,
- handling missing values (imputation or removal),
- detecting and treating outliers,
- encoding categorical variables,
- normalizing or scaling features,
- feature selection and feature engineering.



## Collecting Data from Multiple Sources

Data preparation begins with collecting raw data from a wide variety of sources, including databases, sensors, APIs, web scraping, surveys, and existing public datasets.

During the data collection process, it is important to ensure consistency and compatibility across all sources. Different sources may have different formats, units, naming conventions, or levels of quality. Careful integration, cleaning, and normalization are required to create a unified dataset suitable for analysis or modeling. Proper documentation of sources and collection methods is also essential to maintain reproducibility and data governance.

Public datasets provide an excellent resource for learning, experimentation, and benchmarking. Some widely used datasets across different domains include:
- Tabular datasets: Iris, **Penguins**, Titanic, Boston Housing, Wine, *etc.*
- Image datasets: MNIST, CIFAR-10, CIFAR-100, COCO, ImageNet.
- Text datasets: IMDB Reviews, 20 Newsgroups, Sentiment140.
- Audio datasets: LibriSpeech, UrbanSound8K, ESC-50.
- Video datasets: UCF101, Kinetics, HMDB51.


These datasets are available on platforms like Kaggle, UCI Machine Learning Repository, TensorFlow Datasets, and Hugging Face Datasets, providing accessible resources for practice and innovation.

It should be noted that most of the data available by default is too raw to perform statistical analysis. Proper preprocessing is essential before the data can be used to identify meaningful patterns or to train models for prediction. In the following sections, we use the Penguins dataset as an example to demonstrate essential data preprocessing steps. These include handling missing values, detecting and treating outliers, encoding categorical variables, and performing other necessary transformations to prepare the dataset for ML tasks. Proper preprocessing ensures data quality, reduces bias, and improves the performance and reliability of the models we build.



## The Penguins Dataset

The [Palmer Penguins dataset](https://zenodo.org/records/3960218) is a widely used open dataset in data science and ML education. This dataset contains information on three penguin species that inhabit islands near the Palmer Archipelago in Antarctica: Adelie, Chinstrap, and Gentoo. Each row in the dataset corresponds to a single penguin and records both physical measurements and categorical attributes. The key numerical features include flipper length (mm), culmen length and depth (bill measurements, in mm), and body mass (g). Alongside these, categorical variables such as species, island, and sex are provided.

:::{figure} ./images/4-penguins-categories.png
:align: center
:width: 80%

These data were collected from 2007 - 2009 by Dr. Kristen Gorman with the [Palmer Station Long Term Ecological Research Program](https://lternet.edu/site/palmer-antarctica-lter/), part of the [US Long Term Ecological Research Network](https://lternet.edu/). The data were imported directly from the [Environmental Data Initiative (EDI)](https://edirepository.org/) Data Portal, and are available for use by CC0 license (“No Rights Reserved”) in accordance with the [Palmer Station Data Policy](https://lternet.edu/data-access-policy/).
:::



## Importing Dataset


Seaborn provides the Penguins dataset through its built-in data-loading functions. We can access it using ``sns.load_dataset('penguin')`` and then have a quick look at the data (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/4-Data-Preprocessing.ipynb)):
```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

penguins = sns.load_dataset('penguins')
penguins
```

:::{note}
If you have your own dataset stored in a CSV file, you can easily load it into Python using Pandas with the ``read_csv()`` function. This is one of the most common ways to bring tabular data into a DataFrame for further analysis and processing.

Beyond CSV files, Pandas also supports a wide variety of other file formats, making it a powerful and flexible tool for data handling. For example, you can use ``read_excel()`` to import data from Microsoft Excel spreadsheets, ``read_hdf()`` to work with HDF5 binary stores, and ``read_json()`` to load data from JSON files. Each of these formats also has a corresponding method for saving data back to disk, such as ``to_csv()``, ``to_excel()``, ``to_hdf()``, and ``to_json()``.
:::

:::{csv-table}
:widths: auto
:delim: ;

; species; island; bill_length <br>(mm); bill_depth <br>(mm); flipper <br>length <br>(mm); body_mass <br>(g); sex
0; Adelie; Torgersen; 39.1; 18.7; 181.0; 3750.0; Male
1; Adelie; Torgersen; 39.5; 17.4; 186.0; 3800.0; Female
2; Adelie; Torgersen; 40.3; 18.0; 195.0; 3250.0; Female
3; Adelie; Torgersen; NaN; NaN; NaN; NaN; NaN
4; Adelie; Torgersen; 36.7; 19.3; 193.0; 3450.0; Female
...; ...; ...; ...; ...; ...; ...; ...
339; Gentoo; Biscoe; NaN; NaN; NaN; NaN; NaN
340; Gentoo; Biscoe; 46.8; 14.3; 215.0; 4850.0; Female
341; Gentoo; Biscoe; 50.4; 15.7; 222.0; 5750.0; Male
342; Gentoo; Biscoe; 45.2; 14.8; 212.0; 5200.0; Female
343; Gentoo; Biscoe; 49.9; 16.1; 213.0; 5400.0; Male
:::

There are seven columns include:
- *species*: penguin species (Adelie, Chinstrap, Gentoo)
- *island*: island where the penguin was found (Biscoe, Dream, Torgersen)
- *bill_length_mm*: length of the bill
- *bill_depth_mm*: depth of the bill
- *flipper_length_mm*: length of the flipper
- *body_mass_g*: body mass in grams
- *sex*: male or female

Looking only at the raw numbers in the ``penguins`` DataFrame, or even examining the statistical summaries provided by ``penguins.info()`` and ``penguins.describe()``, often does not give us a clear intuition about the patterns and relationships in the data. To truly understand the dataset, we generally prefer to visualize the data, since graphical representations can reveal trends, groupings, and anomalies that may remain hidden in numerical summaries alone.

One nice visualization for datasets with relatively few attributes is the **Pair Plot**, which can be created using ``sns.pairplot(...)``. It shows a scatterplot of each attribute plotted against each of the other attributes. By using the ``hue='species'`` setting for the pairplot the graphs on the diagonal are layered kernel density estimate plots for the different values of the ``species`` column.

```python
sns.pairplot(penguins[["species", "bill_length_mm", "bill_depth_mm", "flipper_length_mm",
	"body_mass_g"]], hue="species", height=2.0)
```

:::{figure} ./images/4-penguins-pairplot.png
:align: center
:width: 100%
:::


:::{discussion}

Take a look at the pairplot we created. Consider the following questions:
- Is there any class that is easily distinguishable from the others?
- Which combination of attributes shows the best separation for all 3 class labels at once?
- For pairplot with ``hue="sex"``, which combination of features distinguishes the two sexes best?
- What about the one with ``hue="island"``? 
:::



## Handling Missing Values

Upon loading the Penguins dataset into a pandas DataFrame, the initial examination reveals the presence of ``NaN`` (Not a Number) values within several rows (highlighted in the Jupyter notebook). These placeholders explicitly indicate missing or unavailable data for certain measurements, such as bill length or the sex of particular penguins.

Recognizing these missing values is an important first step, as they must be properly handled before performing any data analysis.

```python
penguins_test = pd.concat([penguins.head(5), penguins.tail(5)])
penguins_test.style.highlight_null(color = 'red')
```



### Numerical features


For numerical features such as bill length, bill depth, flipper length, and body mass, several strategies can be applied. A straightforward approach is to **remove any rows with missing values**, but this is often wasteful and reduces the sample size.

A more effective method is imputation: replacing missing numerical values with a suitable estimate. Common choices include the ``mean`` or ``median`` of the feature, depending on the distribution.
Before applying imputation to handle missing numerical values, it is important to first identify where the NaN values occur in the dataset. We can 
- run ``penguins_test.style.highlight_null(color = 'red')``, where NaN values are hightlighed in the output, 
- run ``print(penguins_test.info(), '\n')``, which provide the number of non-null entries in each column, 
	- By comparing the number of non-null entries with the total number of rows, we can quickly identify which features have missing values and how severe the issue is. For example, if the column sex has fewer non-null values than the total number of penguins, we know that sex information is missing for some individuals.
- run ``print(penguins_test.isnull().mean())``, which computes the fraction of missing values in each column, giving us a normalized view of missingness across the dataset.
	- unlike ``.info()``, which only shows counts, this method highlights the relative proportion of missing values, which is particularly helpful when working with large datasets.
	- For instance, the ``.isnull().mean()`` reports that 0.2 (20%) of the entries in ``body_mass_g`` are missing, we can decide whether to impute those values or simply drop the rows without significantly reducing the dataset size.


The next step is to calculate the ``mean`` and ``median`` values for the numerical features. To illustrate this process, we can take the ``body_mass_g`` feature as an example.
```python
body_mass_g_mean = penguins_test.body_mass_g.mean()
body_mass_g_median = penguins_test.body_mass_g.median()

print(f"  mean value of body_mass_g is {body_mass_g_mean}")
print(f"median value of body_mass_g is {body_mass_g_median}")

#   mean value of body_mass_g is 4431.25
# median value of body_mass_g is 4325.00
```

Rather than directly replacing the missing values, we concatenate new columns containing the ``mean`` and ``median`` values for the ``body_mass_g`` feature to the end of the Penguins dataset, and than visualize the distribution of this feature.
```python
penguins_test['BMG_mean'] = penguins_test.body_mass_g.fillna(body_mass_g_mean)
penguins_test['BMG_median'] = penguins_test.body_mass_g.fillna(body_mass_g_median)
```

:::{figure} ./images/4-distribution--body-mass-with-mutations.png
:align: center
:width: 80%
:::


:::{exercise}

How to mutate the missing values with ``mean`` or ``median`` values in place for all numerical values (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/4-Data-Preprocessing.ipynb)).
- for one numerical feature like ``bill_length_mm``?
- for all numerical features in the Penguins dataset?
:::

:::{solution}
- 1. using the following script
	```python
	penguins_test2.loc[penguins_test2["bill_length_mm"].isnull(), "bill_length_mm"] = penguins_test2["bill_length_mm"].mean()
	```
- 2 using the following code
	```python
	# first, select only the numerical columns
	numerical_cols = penguins_test2.select_dtypes(include=['number']).columns
	
	# second, find which of these numerical columns have any missing values 
	numerical_cols_with_nulls = numerical_cols[penguins_test2[numerical_cols].isnull().any()]

	for col in numerical_cols_with_nulls:
		# calculate the mean for the specific column
		col_mean = penguins_test2[col].mean()
		# use `.loc` to replace NaNs only in that specific column with its own mean
		penguins_test2.loc[penguins_test2[col].isnull(), col] = col_mean
	```
:::


In certain scenarios, imputing missing data with values at the **end of distribution** (EoD) of a variable can be a considered strategy. This approach offers the advantage of computational speed and can theoretically capture the significance of missing entries. Typically, the EoD is calculated as an extreme value such as ``mean + 3*std``, where the ``mean`` and ``std`` of the feature can be obtained using the ``.describe()`` method.

However, as demonstrated in the Penguins dataset tutorial, this type of imputation often generates unrealistic values and distorts the original distribution, particularly for features like ``body_mass_g``. Consequently, it can lead to biased analyses and should be used with caution.



### Categorical features


For categorical features such as sex, the approach differs.
- One simple method is to replace missing categories with the most frequent value (``mode``), which assumes the missing value follows the majority distribution.
	```python
	penguins_test.sex.mode()

	penguins_test.fillna({'sex': penguins_test['sex'].mode()[0]}, inplace=True)
	```
- Alternatively, missing values can be treated as a separate category, labeled for example as "Unknown" or "Missing" which allows models to learn if missingness itself carries information.
	```python 
	penguins_test.fillna({'sex': "Missing"}, inplace=True)
	```
- Another option is to apply model-based imputation, where missing categorical values are predicted from other features using classification algorithms.



For all ML tasks we will perform in the following episodes, we adopt a straightforward approach of removing any rows that contain missing values (``penguins_test.dropna()``). This ensures that the dataset is complete and avoids potential errors or biases caused by NaN entries. Although more sophisticated imputation methods exist, dropping rows is a simple and effective strategy when the proportion of missing data is relatively small.

:::{note}
For the other dataset, the strategy for handling missing values is not one-size-fits-all; it depends heavily on whether the missing data is numerical or categorical and the underlying mechanism causing the data to be missing. Ignoring these missing entries, such as by simply dropping the affected rows, may introduce significant bias, reduce statistical power, and ultimately lead to inaccurate or misleading conclusions.
After any imputation, it is essential to perform sanity checks to ensure the imputed values are plausible and to document the methodology transparently. Properly handling missing data in this way transforms an incomplete dataset into a robust and reliable foundation for generating accurate insights and building powerful predictive models.
:::



## Handling Outliers


Outliers are values that are too far from the rest of observations in columns. For instance, if the body mass of most of penguins in the dataset varies between 3000-6000 g, an observation of 7500 g will be considered as an outlier since such an observation occurs rarely.

We obtain the EoD value of the ``body_mass_g`` feature and then check if this value is the outlier for this feature. 
```python
penguins['body_mass_g'].mean() + 3 * penguins_test['body_mass_g'].std()

# EoD value = 7129.0199920504665
```

:::{figure} ./images/4-body-mass-outlier.png
:align: center
:width: 75%
:::


There are several approaches to identify outliers, and one of the most commonly used methods is the **Interquartile Range (IQR)** method. The IQR measures the spread of the middle 50% of the data and is calculated by subtracting the first quartile (25th percentile, Q1) from the third quartile (75th percentile, Q3). Once the IQR is obtained, we can determine the boundaries for detecting outliers. The lower limit is defined as Q1 minus 1.5 times the IQR, and the upper limit is defined as Q3 plus 1.5 times the IQR.
```python
print(f"25% quantile = {penguins_test_BMG_outlier["body_mass_g"].quantile(0.25)}")
print(f"75% quantile = {penguins_test_BMG_outlier["body_mass_g"].quantile(0.75)}\n")

IQR = penguins_test_BMG_outlier["body_mass_g"].quantile(0.75) - penguins_test_BMG_outlier["body_mass_g"].quantile(0.25)
lower_bmg_limit = penguins_test_BMG_outlier["body_mass_g"].quantile(0.25) - (1.5 * IQR)
upper_bmg_limit = penguins_test_BMG_outlier["body_mass_g"].quantile(0.75) + (1.5 * IQR)

print(f"lower limt of IQR = {lower_bmg_limit} and upper limit of IQR = {upper_bmg_limit}")

# 25% quantile = 3550.00
# 75% quantile = 4781.25
# lower limt of IQR = 1703.125 and upper limit of IQR = 6628.125
```


Any data points that fall below the lower limit or above the upper limit are considered outliers, and these points are subsequently removed from the dataset.
```python
penguins_test_BMG_outlier[penguins_test_BMG_outlier["body_mass_g"] > upper_bmg_limit]

penguins_test_BMG_outlier[penguins_test_BMG_outlier["body_mass_g"] < lower_bmg_limit]

penguins_test_BMG_outlier_remove_IQR = penguins_test_BMG_outlier[penguins_test_BMG_outlier["BMG_eod"] < upper_bmg_limit]
```

:::{note}
There are four main techniques for handling outliers in a dataset:
- Remove outliers entirely --- This approach simply deletes the rows containing outlier values, which can be effective if the outliers are errors or rare events that are not relevant to the analysis.
- Treat outliers as missing values --- Outliers can be replaced with NaN and then handled using imputation methods described in previous sections, such as replacing them with the mean, median, or mode.
- Apply discretization or binning --- By grouping numerical values into bins, outliers are included in the tail bins along with other extreme values, which reduces their impact while preserving the overall structure of the data.
- Cap or censor outliers --- Extreme values can be limited to a maximum or minimum threshold, often derived from statistical techniques such as the IQR or standard deviation limits. This approach reduces the influence of outliers without completely removing them from the dataset.
:::


:::{exercise} **The mean–standard deviation approach**

Instead of using the IQR method, the upper and lower thresholds for detecting outliers can also be calculated with the mean-std deviation approach.

In this exercise (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/4-Data-Preprocessing.ipynb)), you will
- Compute the ``mean`` and ``std`` of the ``body_mass_g`` feature.
- Calculate the upper and lower limits for outlier detection using the formulas.
	:::{math}
	lower\_limit = mean - 3.0 \times std

	upper\_limit = mean + 3.0 \times std
	:::
- Identify the outliers and replace them with either the ``mean`` or the ``median`` values of the ``body_mass_g`` feature.
:::

:::{solution}
- ``mean = penguins_test_BMG_outlier["body_mass_g"].mean()`` and ``std = penguins_test_BMG_outlier["body_mass_g"].std()``
- ``lower_bmg_limit = mean - (3.0 * std)`` and ``upper_bmg_limit = mean + (3.0 * std)``
- determination of outliers
	- ``penguins_test_BMG_outlier[penguins_test_BMG_outlier["body_mass_g"] > upper_bmg_limit]`` and ``penguins_test_BMG_outlier[penguins_test_BMG_outlier["body_mass_g"] > upper_bmg_limit]``
- imputation outliers with ``mean`` or ``median``
	- ``penguins_test_BMG_outlier.loc[penguins_test_BMG_outlier["body_mass_g"] < lower_bmg_limit, "body_mass_g"] = mean``
	- ``penguins_test_BMG_outlier.loc[penguins_test_BMG_outlier["body_mass_g"] < lower_bmg_limit, "body_mass_g"] = penguins_test_BMG_outlier["body_mass_g"].median()``
:::



## Encoding Categorical Variables


In the previous sections, we adopted a straightforward approach to handling missing values by simply removing any rows that contained NaN values, whether they were in numerical or categorical features. While this step gives us a cleaner dataset, it is not sufficient on its own to proceed with ML tasks.

The reason is that most ML algorithms are designed to work only with numerical data. They cannot directly process textual or symbolic categories such as species, island, or sex in the Penguins dataset. To make these categorical variables usable in modeling, we need to encode categorical variables into a numerical format while preserving their information.

There are two widely used encoding techniques: **One-hot encoding (OHE)** and **Label Encoding**.



### One-hot encoding


The OHE method creates a new binary column for each category in a feature. For example, the species feature with values {Female, Male, and NaN} would be transformed into three new columns, each indicating the presence (1) or absence (0) of that category for a given row.
```python
from sklearn.preprocessing import OneHotEncoder

penguins_sex = penguins[["species", "island", "sex"]].head(10)

encoder = OneHotEncoder(sparse_output=False)  # `sparse_output=False` to get a dense array
encoded = encoder.fit_transform(penguins_sex[['sex']])

encoded = pd.DataFrame(encoded, columns=["Female", "Male", "NaN"])

penguins_sex_onehotencoding = pd.concat([penguins_sex, encoded], axis=1)
penguins_sex_onehotencoding
```

:::{figure} ./images/4-penguins-sex-ohe-encoding.png
:align: center
:width: 75%
:::


:::{warning}
OHE works well when the number of categories is small and when categories are unordered. However, it can lead to very large datasets if a feature contains many unique categories (high cardinality).
:::



### Label encoding

Label encoding transforms a categorical feature into a single numerical column by assigning a unique integer to each category in a feature. For example, the sex feature with values {Female, Male, NaN} could be encoded as Female = 0, Male = 1, and missing values handled separately or imputed beforehand (in our case, NaN is encoded as  NaN = 2).

Unlike one-hot encoding, which creates multiple columns, label encoding produces only one column, making it more memory-efficient. However, it introduces an artificial ordinal relationship between categories (*e.g.*, implying Dream > Biscoe), which may not be meaningful and can affect algorithms that assume numeric order matters, such as linear regression or distance-based methods.
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
encoded = encoder.fit_transform(penguins_sex['sex'])
encoded = pd.DataFrame(encoded, columns=["sex_LE"])

penguins_sex_labelencoding = pd.concat([penguins_sex, encoded], axis=1)
```


Beyond these two common methods, there are several other encoding strategies, especially useful for more complex datasets:
- **Ordinal Encoding**: A variation of label encoding designed specifically for variables that have a natural and meaningful order (*e.g.*, Small = 0, Medium = 1, Large = 2).
- **Binary Encoding**: Converts categories into binary code, achieving a good compromise between one-hot and label encoding for high-cardinality data.
- **Target Encoding** (Mean Encoding): Replaces each category with the average value of the target variable for that category. It is powerful but prone to overfitting if not carefully implemented.
- **Frequency Encoding**: Replaces categories with their frequency of occurrence in the dataset. This can be useful for dealing with high-cardinality features.

The choice of encoding method is a consequential modeling decision. It should be guided by the type of categorical data (nominal *vs.* ordinal), the number of unique categories, the type of ML algorithm being used. A proper encoding ensures that categorical features are accurately represented in numerical form, allowing algorithms to learn patterns effectively without introducing bias or noise.



### The ``get_dummies()`` function in Pandas

In addition to OHE and LE, we can also use the ``.get_dummies()`` in Pandas to handle categorical variables by converting them into a one-hot encoded (dummy variable) format.
```python
dummy_encoded = pd.get_dummies(penguins_sex['sex']).astype(np.int8)
penguins_sex_dummyencoding = pd.concat([penguins_sex, dummy_encoded], axis=1)
```

:::{note}
This function ignores ``NaN`` values by default (it simply leaves them out).
:::



## Feature Engineering


Feature engineering is a part of the broader data processing pipeline in ML workflows. It involves using domain knowledge to select, modify, or create new features --- variables or attributes --- from existing data to help algorithms better understand patterns and relationships.

Feature engineering is crucial because the quality of features directly impacts a model's predictive power. Well-crafted features can simplify complex patterns, reduce overfitting, and improve model interpretability, leading to better generalization and performance on unseen data. By tailoring features to the problem at hand, feature engineering bridges the gap between raw data and actionable insights, often making the difference between a mediocre and a high-performing model.

Feature engineering is closely related to data preprocessing, but they serve different purposes.
- Data processing (or data preprocessing) is about cleaning and preparing data --- handling missing values, removing duplicates, correcting data types, and ensuring consistency. This step makes the data **usable**.
- Feature engineering, on the other hand, comes after basic processing and focuses on improving the predictive power of dataset.
- In essence, **data preprocessing ensures data quality**, while **feature engineering enhances data value** for ML models.
- Both are essential steps in building effective and accurate predictive systems.



:::{keypoints}
- Data Preparation is a critical, foundational step in the machine learning workflow.
- Introduction to the Palmer Penguins dataset, and demonstration of loading the dataset into a pandas DataFrame for analysis.
- Statistical analysis and Visualization of the Penguins dataset using Pandas and Seaborn.
- Identification and handling missing numerical values and outliers.
- Convertion of categorical features into numerical formats.
- Feature engineering is the process of creating, transforming, or selecting the right features from raw data to improve the performance of machine learning models.
:::
