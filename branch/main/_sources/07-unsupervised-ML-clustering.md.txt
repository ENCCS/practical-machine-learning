# Unsupervised Learning (I): Clustering



:::{objectives}
- Explain what unsupervised learning is and how clustering fits into it.
- Understand main ideas behind centroid-based (K-Means), hierarchical, density-based (DBSCAN), model-based (GMM), and graph-based (Spectral Clustering) methods.
- Apply representative clustering algorithms in practice.
- Understand how to evaluate clustering quality using confusion matrices, silhouette scores, or visualizations.
:::



:::{instructor-note}
- 40 min teaching/demonstration
- 40 min exercises
:::



## Unsupervised Learning


In [Episode 5](./05-supervised-ML-classification.md) and [Episode 6](./06-supervised-ML-regression.md), we have explored supervised learning, where each training example includes both input features and the corresponding output.
This setup enables the models to learn a direct mapping from inputs to targets. For the Penguins dataset, both classification and regression models, such as logistic/linear regression, decision trees, and neural networks, were applied to either classify penguin species or predict body mass based on flipper length.


It is important to emphasize that **supervised learning** depends heavily on labeled data, which may not always be available in real-world scenarios. Collecting labeled data can be expensive, time-consuming, or even impossible.
In such cases, we turn to **unsupervised learning** to uncover patterns and structure in the data.

In unsupervised learning, the dataset contains only the input features without associated labels. The goal is to discover hidden patterns, structures within the data, and derive insights without explicit supervision.
Unsupervised learning is essential for analyzing the vast amounts of raw data generated in real-world applications, from scientific research to business intelligence. Its significance can be seen across several key areas:
- **Exploratory Data Analysis** (EDA): Techniques such as clustering and dimensionality reduction are fundamental for understanding the structure of complex datasets. They can reveal natural groupings, trends, and correlations that might otherwise remain hidden, providing a crucial first step in any data-driven investigation.
- **Anomaly Detection**: Unsupervised learning is vital for maintaining security and operational integrity. By modeling "normal" behavior, algorithms can identify unusual patterns, such as fraudulent financial transactions, network intrusions, or rare mechanical failures, without needing labeled examples of every type of anomaly.
- **Feature Engineering** and **Representation Learning**: Methods like **Principal Component Analysis** (PCA) can compress data into its most informative components, reducing noise and improving the efficiency and performance of downstream supervised models.


In this and the following episodes, we will apply **Clustering** and **Dimensionality Reduction** methods on the Penguins dataset to explore its underlying structure and uncover hidden patterns without the guidance of pre-existing labels. 
By employing clustering methods like K-means, we aim to identify species-specific clusters or other biologically meaningful subgroups among Adelie, Gentoo, and Chinstrap penguins. Additionally, dimensionality reduction techniques like PCA will simplify the dataset’s feature space, enabling visualization of complex relationships and enhancing subsequent analyses.
These approaches will deepen our understanding of penguin characteristics, reveal outliers, and complement supervised methods by providing a robust framework for exploratory data analysis.



## Clustering


Clustering is one of the most widely used techniques in unsupervised learning, where the goal is to group similar data points together without using predefined labels. For example, if we cluster penguins based on their physical characteristics such as flipper length, body mass, bill length, and bill depth, we may be able to separate them into natural groups that correspond to their species -- even without explicitly providing species labels.


Clustering, however, presents several fundamental challenges. One major issue is determining the optimal number of clusters (*k*), which is often non-trivial. Many algorithms, such as k-means, require specifying the number of clusters in advance, which may not be immediately obvious from the data.

As illustrated in the figure below, the data could be grouped into two, three, or four clusters, depending on the level of granularity chosen. Selecting too few clusters may oversimplify the structure and miss important patterns, while choosing too many clusters can lead to overfitting, where random noise is mistakenly treated as meaningful groups.


:::{figure} ./images/7-rawdata-into-clusters.png
:align: center
:width: 80%
:::


In addition, the quality and scale of features also have a significant impact on clustering results. Features with larger scales can dominate distance computations, making preprocessing steps such as standardization essential.

It should be emphasized that the interpretation of clustering results is subjective. While an algorithm can identify groups, it is up to the analyst to determine whether those groups are meaningful or merely artifacts of the algorithm.

Clustering outcomes are also highly sensitive to the choice of algorithm and distance metric. For instance, K-Means tends to find spherical clusters, whereas DBSCAN (Density-Based Spatial Clustering of Applications with Noise) can detect arbitrarily shaped clusters and identify outliers, may leading to very different conclusions using the same dataset.

In this episode, we will apply multiple clustering algorithms to evaluate their performance on the Penguins dataset.



## Data Preparation


Following the procedures used in previous episodes, we apply the same preprocessing steps to the Penguins dataset, including handling missing values and detecting outliers. For the clustering task, categorical features are not required, so encoding them is unnecessary.



## Data Processing


The data processing is straightforward for the clustering task: we simply extract the numerical variables and apply standardization.

```python
penguins = sns.load_dataset('penguins')
penguins_clustering = penguins.dropna()

X = penguins_clustering[['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']]

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```



## Training Model & Evaluating Model Performance



### K-Means Clustering


K-Means clustering, a centroid-based partitioning method, is a widely used unsupervised learning algorithm that divides data into k distinct, non-overlapping clusters. K-Means operates on a simple yet powerful principle: each cluster is represented by its centroid, which is the mean position of all points within the cluster.

The algorithm begins by randomly initializing k centroids in the feature space and then iteratively refines their positions through a two-step **expectation**-**maximization** process:
- In the expectation step, each data point is assigned to its nearest centroid based on Euclidean distance, forming preliminary clusters.
- In the maximization step, the centroids are recalculated as the mean of all points assigned to each cluster. This process repeats until convergence, typically when centroid positions stabilize or cluster assignments no longer change significantly.


:::{figure} ./images/7-kmeans-description-expectation-maximization.png
align: center
width: 100%
:::


We build a ``kmeans`` model using the ``KMeans`` class from ``sklearn.cluster`` with specified parameters. By fitting the constructed ``kmeans`` model, we can obtain the cluster ID to which each point belongs.
- ``n_clusters=k``, number of clusters to find from the dataset
- ``n_init=10``, number of times KMeans runs with different centroid seeds (default 10 or more)
- ``random_state``, ensures reproducibility

```python
from sklearn.cluster import KMeans

k = 3  # we know there are 3 species
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)
```

Evaluating clustering efficiency can be tricky because, unlike supervised learning, we don’t always have "true labels". Depending on whether we have ground truth or not, there are two main evaluation approaches:
- If we don't know the actual labels, we can measure how well each data point fits within its assigned cluster compared to other clusters, such as the **Silhouette Score**.
- If we know the actual labels (*e.g.*, penguin species), we can measure how well clustering recovers them, such as the **Adjusted Rand Index** (ARI).

Here we adopte these two matrics to evaluate model performance.
```python
penguins_cluster = penguins.dropna().copy()
penguins_cluster['cluster'] = clusters

from sklearn.metrics import silhouette_score, adjusted_rand_score

sil_score = silhouette_score(X_scaled, clusters)
ari_score = adjusted_rand_score(penguins_cluster['species'], clusters)

print(f"Silhouette Score: {sil_score:.3f}")
print(f"Adjusted Rand Index (vs true species): {ari_score:.3f}")
```

Higher ARI values indicate that the clustering results align closely with the true groupings. An ARI of 1.0 represents perfect agreement, and of 0 corresponds to random clustering. Negative ARI values suggest clustering performance worse than random chance.


The Silhouette Score ranges between -1 to +1.
- A score of +1 indicates that samples in the dataset are well-matched to their own cluster and clearly separated from other clusters.
- A score of 0 suggests that samples lie on the boundary between clusters.
- A score of -1 implies that samples may have been incorrectly assigned to clusters.


We take a further step to visualize the distributions of penguins by comparing their true labels with the clusters determined by the ``kmeans`` model.

:::{figure} ./images/7-kmeans-penguins-label-cluster.png
:align: center
:width: 100%
:::


We have 333 penguins, and from the plots above it is difficult to determine how many penguins belong to each cluster and what their species are. This can be clarified by examining the distribution of penguin species across clusters and computing a cross-tabulation of two categorical variables using the ``.crosstab()`` method in Pandas.

```python
cross_tab = pd.crosstab(penguins_cluster['cluster'], penguins_cluster['species'])

cross_tab.plot(kind='bar', stacked=True, figsize=(9, 6))
```

:::{figure} ./images/7-kmeans-penguins-in-clusters.png
:align: center
:width: 80%
:::



**Determination of optimal number of clusters**


At the beginning of this section, we set ``n_clusters = 3``, because we already knew that the Penguins dataset contains three species. However, in many real-world applications, the true number of groups or clusters is not known in advance. In such cases, it becomes essential to estimate the appropriate number of clusters before performing the actual clustering task.

To address this, we employ two widely-used heuristic methods, the **Elbow Method** and the **Silhouette Score analysis**, to determine the optimal cluster number *k*.
- The Elbow Method quantifies the quality of the clustering using **Within-Cluster Sum of Squares** (WCSS), which measures how tightly the data points in each cluster are grouped around their centroid.
	- Intuitively, we want clusters that are tight and cohesive, which corresponds to a low WCSS.
	- As we increase the number of clusters *k*, the WCSS will always decrease because the clusters become smaller and tighter. Beyond a certain point, the improvement of k becomes marginal contribution to WCSS.
	- By plotting the WCSS against the number of clusters, we look for the **elbow** point in the curve, which represents a good balance between model complexity and cluster compactness.
- The Silhouette Score Method evaluates the quality of clustering by measuring how similar each data point is to its own cluster compared to other clusters.
	- For a single data point, the silhouette coefficient compares the average distance to points in its own cluster (cohesion) to the average distance to points in the nearest neighboring cluster (separation).



We rerun the computation using the K-Means algorithm across a range of cluster values. For each tested number of clusters, we compute both the WCSS and the Silhouette Score. By plotting these metrics against the number of clusters *k*, we can visually assess the trade-offs and identify the most suitable cluster count. The code example and corresponding output are shown below.

```python
max_clusters = 15
wcss = []
silhouette_scores = []

for i in range(2, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    wcss.append(kmeans.inertia_)

    silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette_avg)
    
    print(f"Clusters: {i}, WCSS: {kmeans.inertia_:.2f}, Silhouette: {silhouette_avg:.3f}")
```

:::{figure} ./images/7-kmeans-optimal-parameter.png
:align: center
:width:100%
:::



:::{figure} ./images/7-kmeans-234-clusters.png
:align: center
:width:100%
:::



:::{discussion} Why does K-Means suggest grouping the penguins into 2 clusters?

K-Means suggesting 2 clusters instead of 3 in the Penguins dataset is actually a common outcome, and it happens for several reasons:
- feature overlap between species:
	- Some penguin species, like Adelie and Chinstrap, have very similar measurements for features such as bill length, bill depth, flipper length, and body mass.
	- K-Means uses Euclidean distance, so if two species’ points are close in feature space, the algorithm may group them into a single cluster.
- data scaling or feature selection:
	- Features with larger scales or high correlation can dominate the distance calculation.
	- If preprocessing is not optimal, K-Means may prioritize grouping based on dominant features rather than species distinctions.
- K-Means assumes spherical clusters:
	- K-Means works best when clusters are roughly spherical and equally sized.
	- If clusters have different shapes, densities, or overlap, K-Means may merge two clusters to minimize WCSS, resulting in fewer clusters than the actual number of species.
- Elbow or Silhouette methods suggest 2:
	- When using the elbow method, the WCSS curve may show a clear "elbow" at k=2, indicating that adding a third cluster doesn’t significantly reduce WCSS.
	- Similarly, the average Silhouette Score might be highest for k=2, because splitting the overlapping species into separate clusters reduces cohesion.
:::



### Hierarchical Clustering


**Hierarchical clustering** is an unsupervised learning method that builds a hierarchy of clusters by either divisive (top-down) or agglomerative (bottom-up) strategies. In the agglomerative approach, each data point starts as its own cluster, and the algorithm iteratively merges the closest clusters based on a distance metric. This continues until all points are merged into a single cluster. The result can be visualized as a **dendrogram**, a tree-like diagram showing the nested structure of clusters at different levels of granularity.


:::{callout} Hierarchical Clustering *vs.* Decision Tree

Hierarchical clustering is conceptually similar to a decision tree in some ways, but it is not the same as a decision tree or random forest.
- Similarity is that both build a tree-like structure
- Key differences
	- Purpose of the tree
		- In hierarchical clustering, the tree (dendrogram) represents nested clusters and shows the order in which points/clusters are merged or split.
		- In decision trees, the tree represents decision rules to predict a target variable.
	- Supervised *vs.* unsupervised algorithms 
	- With and without ensemble concept
		- Random forest is an ensemble of decision trees and focuses on improving prediction accuracy and reducing overfitting.
		- Hierarchical clustering has no ensemble concept or predictive objective; it is purely descriptive.

In short, **Hierarchical Clustering is structurally similar to a tree (like a dendrogram)**, but **it is unsupervised and descriptive, unlike Decision Trees or Random Forests, which are supervised and predictive**.
:::



We first use SciPy and then Scikit-Learn packages for the clustering task, for the purpose of comparison.
In SciPy, hierarchical clustering involves two steps: **computing the linkage matrix** (``linkage()``), and then **extracting clusters from it** (``fcluster()``). In the code listed below, 
- ``linkage()`` computes the full hierarchical clustering tree (linkage matrix), storing all merges, distances, and cluster sizes.
- ``fcluster()``, cuts the dendrogram at a specified threshold to produce a flat clustering, here forming 3 clusters (``t=3``, criterion='maxclust').

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# compute linkage matrix
linked = linkage(X_scaled, method='ward')

# assign 3 clusters based on dendrogram cut
labels_scipy = fcluster(linked, t=3, criterion='maxclust')

penguins_cluster = penguins.dropna().copy()
penguins_cluster['hier_cluster_scipy'] = labels_scipy
```


Next we plot the dendrogram to visualize clustering structure.

:::{figure} ./images/7-hierarcical-dendrogram.png
:align: center
:width: 100%
:::



Here, we move to the Scikit-learn package and employ ``AgglomerativeClustering`` to construct the clustering model with hyperparameters.
- The parameter ``linkage`` determines how the distance between clusters is calculated. There are several options for this parameters:
	- ``ward``,  minimizes the variance of merged clusters (only works with Euclidean distance).
	- ``complete``, maximum distance between points in clusters.
	- ``average``, average distance between points in clusters.
	- ``single``, minimum distance between points in clusters.
- The parameter ``metric`` is the distance metric used to compute the distance between points.
	- ``euclidean`` is the standard straight-line distance in feature space.
	- There are also other options like ``manhattan``, ``cosine``, *etc.*

```python
from sklearn.cluster import AgglomerativeClustering

hc = AgglomerativeClustering(n_clusters=3, metric='euclidean', linkage='ward')
labels = hc.fit_predict(X_scaled)

penguins_cluster['hier_cluster_aggl'] = labels
```


We can examine the number of penguins in each species within the clusters determined by the two models, and visualize their distributions using a confusion matrix.

:::{figure} ./images/7-hierarchical-clusters-from-scipy-scikit-learn.png
:align: center
:width: 100%
:::

:::{figure} ./images/7-hierarchical-confusion-matrix.png
:align: center
:width: 100%
:::



### DBSCAN


**DBSCAN** (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed while marking points in low-density regions as outliers. Unlike K-Means, DBSCAN does not require specifying the number of clusters in advance, making it particularly useful when the number of natural clusters is unknown. It is also capable of identifying clusters of arbitrary shapes, unlike centroid-based methods that assume roughly spherical clusters. This makes DBSCAN robust to clusters with irregular shapes or varying sizes.

We build a model using the ``DBSCAN`` class from ``sklearn.cluster`` with specified parameters. DBSCAN relies on two key parameters:
- ``eps``, the radius that defines the neighborhood around a point.
- ``min_samples``, the minimum number of points required within a point’s eps neighborhood for it to be considered a core point.
- Below we use ``eps=0.55`` and ``min_samples=5`` in the code.
	- You can experiment with other ``eps`` values (*e.g.*, 0.50 and 0.80) while keeping ``min_samples=5`` to observe how the clustering results change.

```python
from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.55, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# evaluate clustering (only if at least 2 clusters found)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print(f"DBSCAN found {n_clusters} clusters (and {sum(labels==-1)} noise points).")

if n_clusters > 1:
    sil_score = silhouette_score(X_scaled, labels)  # FIXED
    ari_score = adjusted_rand_score(penguins_cluster['species'], labels)
    print(f"Silhouette Score: {sil_score:.3f}")
    print(f"Adjusted Rand Index (vs true species): {ari_score:.3f}")
```


:::{note}

DBSCAN classifies samples into three types:
- **core points**: points with at least ``min_samples`` neighbors within ``eps``.
- **border points**: points within ``eps`` of a core point but with fewer than ``min_samples`` neighbors themselves.
- **noise points** (outliers): points that are neither core nor border points.
:::


Next, we visualize the distributions of penguins in each cluster, including any points identified as noise.

:::{figure} ./images/7-dbscan-point-types.png
:align: center
:width: 80%
:::


We further examine the distribution of penguin species across clusters using the cross-tabulation (``.crosstab()``) method in Pandas.

```python
cross_tab = pd.crosstab(penguins_cluster['dbscan_cluster'], penguins_cluster['species'])

cross_tab.plot(kind='bar', stacked=True, figsize=(9,6))
```

:::{figure} ./images/7-dbscan-penguins-in-clusters.png
:align: center
:width: 80%
:::


:::{exercise}

In this exercise (code examples are availalbe in the [Jupyter Notebook](./jupyter-notebooks/7-ML-Clustering.ipynb), we will
- Experiment with ``eps`` values (0.50 and 0.80) while keeping ``min_samples=5`` to observe how the clustering results change.
- Computations with more combinations of ``eps`` and ``min_samples``.
- Explore methods to find optimal hyperparameters (using grid search and cross-validation).
:::



### Gaussian Mixture Models


After exploring centroid-based methods like K-Means, hierarchical clustering models, and density-based approaches such as DBSCAN, we now turn our attention to model-based clustering algorithms.
Unlike the previous methods that rely primarily on distance metrics or density thresholds, model-based clustering assumes that the data come from a mixture of underlying probability distributions. Each distribution corresponds to a cluster, and the algorithm tries to estimate both the parameters of the distributions and the clusters.

Since model-based clustering assumes that a dataset is generated from a mixture of underlying probability distributions, a variety of algorithms have been developed to handle different types of distributions. The choice of model depends on the nature of the data.
- When dealing with continuous numerical features that approximately follow a bell-shaped distribution, **Gaussian Mixture Models** (GMMs) are the most common choice.
- If the data consists of count values, mixture models based on Poisson distributions can be used.
- For categorical data, methods like Latent Class Analysis (LCA), which treats clusters as latent categorical variables, are often applied.
- In more flexible Bayesian frameworks, Dirichlet Process Mixture Models allow the number of clusters to be inferred directly from the data, avoiding the need to predefine it.


The GMM assumes that data points are generated from a mixture of Gaussian distributions, each representing a cluster. Instead of assigning points strictly to one cluster (like K-Means), GMM assigns each point a probability of belonging to each cluster, making it a soft clustering method.


In the following example, we construct the GMM model with the specified hyperparameters.
- ``n_components=3`` means the number of Gaussian distributions (clusters) to fit
- ``covariance_type`` controls the form of the covariance matrix for each Gaussian distribution
	- ``full`` means that each cluster has its own general covariance matrix (most flexible, allows ellipsoidal shapes).
	- other options like ``tied``, ``diag``, and ``spherical`` corresponding to clusters with different shapes

```python
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score

# build GMM model with 3 components (clusters)
gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
gmm.fit(X_scaled)
labels_gmm = gmm.predict(X_scaled)
```



Following the steps in the [Jupyter Notebook](./jupyter-notebooks/7-ML-Clustering.ipynb), we can 1) examine the distribution of penguin species across clusters using the cross-tabulation method, 2) visualize the distribution of penguins within each cluster, and 3) illustrate their distributions with a confusion matrix.


Here, we specifically highlight the distributions penguins data points in clusters and the shapes clusters obtained from the KMeans and GMM models.

:::{figure} ./images/7-gmm-elliptical-clusters.png
:align: center
:width: 100%
:::



### Spectral Clustering

Following centroid-based, density-based, and model-based methods, we now turn our attention to **Spectral Clustering** algorithms.

Spectral Clustering represents a fundamentally different approach: rather than relying purely on distances between points or density, it leverages graph theory and the eigenstructure of similarity matrices to uncover clusters. That is, the main idea of this method is to represent the dataset as a graph where each node is a data point and edges encode the similarity between points (*e.g.*, using a Gaussian kernel). By computing the eigenvectors of the graph Laplacian, the algorithm transforms the original data into a lower-dimensional space where clusters become more distinguishable. Once in this space, standard clustering techniques, such as K-Means, are applied to assign cluster labels.

This method is especially powerful for datasets with complex, **non-convex** cluster shapes, where traditional algorithms like K-Means or Hierarchical Clustering may fail to capture the true underlying structure.


We adopted similar procedures to build the model, examine the distribution of penguin species across clusters using the cross-tabulation method, and visualize the distribution of penguins within each cluster.

```python
from sklearn.cluster import SpectralClustering

# build model using Spectral Clustering (graph-based)
spectral = SpectralClustering(
    n_clusters=3,
    affinity='rbf',   # Gaussian kernel
    gamma=1.0,        # controls width of the Gaussian
    assign_labels='kmeans',
    random_state=42
)
labels = spectral.fit_predict(X_scaled)
penguins_cluster['spectral_cluster'] = labels

# evaluate clustering
ari = adjusted_rand_score(penguins_cluster['species'], labels)
sil = silhouette_score(X_scaled, labels)
print(f"Adjusted Rand Index (vs species): {ari:.3f}")
print(f"Silhouette Score: {sil:.3f}")
```

:::{figure} ./images/7-spectral-confusion-matrix-kmeans-gmm.png
:align: center
:width: 100%
:::


:::{callout} Spectral Clustering *vs.* Gaussian Mixture Models *vs.* K-Means
:class: dropdown

From the confusion matrix shown above, it seems that Spectral Clustering and K-Means models are less effective than the GMM model on the Penguins dataset. Main reasons may attribute to:
- Small dataset size:
	- The Penguins dataset has only 333 samples.
	- Spectral clustering computes the eigenvectors of the similarity matrix, which can be less stable with small datasets, leading to variability in cluster assignments.
- Small number of features/low dimensionality
	- The Penguins dataset typically uses only 4 numerical features. In such low-dimensional, fairly well-separated data, simpler methods like K-Means or Gaussian Mixture Models often perform just as well or better.
	- Spectral clustering shines when clusters are non-convex or complexly shaped in high-dimensional spaces.
:::


::::{exercise}

Here, we apply these two models to the classic **two-moon dataset**, a well-known synthetic dataset with non-linearly separable clusters. This allows us to visually and quantitatively evaluate how each algorithm performs in capturing complex, non-convex structures and to compare their strengths and limitations in a controlled setting.

:::{figure} ./images/7-spectral-kmeans-two-moon-dataset.png
:align: center
:width: 80%
:::

**Spectral Clustering excels for datasets with complex, non-convex cluster shapes, where traditional algorithms like K-Meansmay fail to capture the true underlying structure**.
::::


## Comparison of Clustering Models

:::{figure} ./images/7-comparison-sil-ari-scores.png
:align: center
:width: 100%
:::

| Method | Type/Approach | Key Characteristics | Pros | Limitations |
| :----: | :----: | :----: | :----: | :----: |
| K-Means | Centroid-based | Partitions data into <br>k clusters by minimizing <br>within-cluster variance; <br>clusters represented by centroids | Simple, fast, widely used; <br>interpretable | Assumes spherical clusters; <br>sensitive to initialization and outliers; <br>requires specifying k |
| Hierarchical Clustering <br>(SciPy) | | Similar to Scikit-Learn, <br>uses linkage matrix <br>and fcluster to assign clusters | Flexible; supports different <br>distance metrics and linkage methods | Requires careful selection of <br>threshold to cut dendrogram; <br>can be slow for large data |
| Hierarchical Clustering <br>(Scikit-Learn) | | Builds a hierarchy of clusters <br>either bottom-up (agglomerative) <br>or top-down (divisive); <br>linkage criteria define <br>merge/split decisions | Dendrogram visualization; <br>no need to pre-specify number of clusters | Computationally expensive for large datasets; <br>choice of linkage affects results |
| DBSCAN | Density-based | Groups points <br>based on density; identifies core, border, <br>and noise points; no need to specify <br>number of clusters | Detects arbitrarily shaped clusters; <br>robust to outliers; <br>identifies noise | Sensitive to eps and min_samples; <br>struggles with varying densities |
| GMM | Model-based | Assumes data generated <br>from a mixture of Gaussian distributions; <br>each cluster has mean, covariance, and weight | Can model elliptical clusters; <br>provides probabilities for cluster membership | Sensitive to initialization; <br>may converge to local optima; <br>assumes Gaussian distribution |
| Spectral Clustering | Graph-based | Uses graph Laplacian of similarity matrix; <br>clusters derived from eigenvectors; <br>can handle non-convex shapes | Captures complex structures; <br>good for connected or non-spherical clusters | Computationally expensive for large datasets; <br>sensitive to affinity and connectivity; <br>may fail on disconnected graphs |



:::{keypoints}
- Clustering is about grouping data points that are similar to each other, without using labels.
- Representative algorithms for clustering tasks: K-Means (centroid-based), Hierarchical, DBSCAN (density-based), Gaussian Mixture Models (model-based), and Spectral Clustering (graph-based).
- Use tools like Silhouette score or visual plots to see how well clusters are separated.
- **No single algorithm is "best", and the right method depends on your data: size, type, shape, and what you want to achieve**.
:::

