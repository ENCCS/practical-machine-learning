# Fundamentals of Machine Learning



:::{objectives}
- Describe the representative types of machine learning (supervised, unsupervised, semi-supervised, reinforcement learning).
- Explain the general workflow of a machine learning project.
- Introduce representative ML libraries and discuss their pros and cons.
:::



:::{instructor-note}
- 25 min teaching
- 0 min exercising
:::



## Types of Machine Learning


ML can be broadly categorized into three main types depending on how the models learn from input data and the nature of the input data they process.

:::{figure} ./images/2-ML-three-types.png
:align: center
:width: 80%

Three main types of machine learning. Main approaches include classification and regression under the supervised learning and clustering under the unsupervised learning. Reinforcement learning enhance the model performance by interacting with environment. Coloured dots and triangles represent the training data. Yellow stars represent the new data which can be predicted by the trained model. This figure was taken from the paper [Machine Learning Techniques for Personalised Medicine Approaches in Immune-Mediated Chronic Inflammatory Diseases: Applications and Challenges ](https://www.frontiersin.org/journals/pharmacology/articles/10.3389/fphar.2021.720694/full).
:::



### Supervised learning


In supervised learning, the model is trained on a labeled dataset, where each input is paired with a corresponding output (label). The goal is to learn a mapping from inputs to outputs to make predictions on new, unseen data.

Supervised learning has two subtypes: **Classification** (predicting discrete categories) and **Regression** (predicting continuous values).

Here are representative examples of these two subtypes in real-word problems:
- **Classification**: email spam detection (spam/ham), image recognition (cat/dog), medical diagnosis (disease/no disease).
- **Regression**: house price prediction, weather forecasting.



### Unsupervised learning


In unsupervised learning, the model works with unlabeled data, identifying patterns, structures, or relationships within the data without explicit guidance on what to predict.

Unsupervised learning also has two subtypes: **Clustering** (grouping similar data points together) and **Dimensionality Reduction** (simplifying data by reducing features while preserving important information)

Representative examples of these two subtypes in real-word problems:
- **Clustering**: customer segmentation in marketing (grouping users by behavior), image segmentation (grouping similar pixels).
- **Dimensionality Reduction**: compressing high-dimensional data (*e.g.*, reducing image features for faster processing), anomaly detection.



### Reinforcement learning


The model (agent) learns by interacting with an environment. It takes actions, receives feedback (rewards or penalties), and learns a strategy (policy) to maximize long-term rewards.

Representative examples of reinforcement learning in real-word problems: game-playing AI (*e.g.*, AlphaGo), robot navigation, autonomous driving.



### Other subtypes


In addition to supervised and unsupervised learning, there are other important paradigms in machine learning.
- **Semi-supervised learning** bridges the gap between supervised and unsupervised learning by using a small amount of labeled data together with a large amount of unlabeled data, helping models learn more effectively when labeling is expensive or time-consuming (*e.g.*, medical image analysis).
- **Self-supervised learning** is a form of unsupervised learning where the model generates its own labels from the data -- typically for pretraining models on tasks like image or language understanding, enabling them to learn robust representations without explicit labels (*e.g.*, predicting the next word in a sentence, and filling in missing image patches)
- **Transfer learning** involves applying knowledge from a pretrained model, trained on a large, general dataset, to a new, related task, significantly reducing training time and data requirements (*e.g.*, fine-tuning a speech recognition model for a new dialect).

These techniques expand the capabilities and versatility of machine learning across data-limited or computationally constrained environments.



## Machine Learning Workflow



### What is a workflow for ML?


A machine learning workflow is a structured approach for developing, training, evaluating, and deploying machine learning models. It typically involves several key phases, including data collection, preprocessing, model training and evaluation, and finally, deployment to production.

Here is a graphical representation of ML workflow, and a concise overview of the key steps are described below.

:::{figure} ./images/2-ML-workflow.png
:align: center
:width: 100%
:::



### Problem definition and project setup


**Problem Definition** is the first and most critical phase of any ML project. It sets the direction, scope, and goals for the entire project.
- we should understand the problem domain: what is the real-world problem we are trying to solve? are we predicting, classifying, or grouping data? (*e.g.*, predict house prices, detect spam emails, cluster customers)
- we should determine if ML is the appropriate solution for the problem
- we then should identify the expected outputs: what will the ML model produce? (*e.g.*, a number, a label, or a probability)
- we define the type of ML task (*e.g.*, classification and regression tasks for supervised learning, clustering, dimensionality reduction for unsupervised learning, and decision-making tasks for reinforcement learning)


**Project Setup** is to set up the programming/development environment for the project.
- hardware requirements (CPU, SSD, GPU, cloud platforms, *etc.*)
- software requirements (programming languages and libraries, ML/DL frameworks, and development tools, IDEs, Git/Docker, *etc.)
- project structure: organize your project for clarity and scalability

A typical ML project structure looks like this:
```
  ML_Project/
  ├── data/                 # raw and processed data
  │   ├── raw/              # original, unprocessed data
  │   ├── processed/        # cleaned, preprocessed data
  ├── notebooks/            # jupyter notebooks for EDA & modeling
  ├── src/                  # source code
  │   ├── utils/            # utility functions (*e.g.*, metrics, logging)
  │   ├── preprocessing.py  # data cleaning script  
  │   └── train.py          # model training script
  ├── models/               # trained model files (*e.g.*, .pkl, .h5)
  ├── tests/                # unit and integration tests
  ├── README.md             # project overview and setup instructions
  ├── requirements.txt      # project dependencies
  ├── config.yaml           # configuration file for hyperparameters and paths
```



### Data collection and preprocessing


In ML, data collection and preprocessing are crucial steps that significantly affect the performance of a model. High-quality, well-processed data leads to better predictions, while poor data can result in unreliable models.
- **data collection**: Gather the necessary data from various sources (*e.g.*, databases, APIs (twitter, linkedin, *etc.*), or manual collection), and ensure that data is representative and sufficient for the problem.
- **data preprocessing**: Clean and prepare data by handling missing values (drop, impute, or predict), removing duplicates or irrelevant data, fixing inconsistencies (*e.g.*, "USA" vs. "United States"), normalizing/scaling features, encoding categorical variables, and addressing outliers, and other data quality issues.
- **exploratory data analysis** (EDA): Analyze data to uncover distributions, correlations, patterns, anomalies, and insights using visualizations and statistical methods. This helps in feature selection and understanding data distribution.
- **feature engineering**: Create or select relevant features to improve model performance. This may involve dimensionality reduction (*e.g.*, PCA (principal component analysis)) or creating new features based on domain knowledge.
- **data splitting**: Divide the dataset into training, validation, and test sets (*e.g.*, 70-15-15 split) to evaluate model performance and prevent overfitting.



### Model selection and training


Model Selection and Training refer to the process of choosing an appropriate model architecture and training it to learn patterns from data to solve a specific task. It involves selecting the appropriate algorithms (*e.g.*, linear/logistic regression, decision trees, neural networks, Gradient Boosting) based on the problem type, configuring its hyperparameters, and optimizing its parameters using training data to minimize error or maximize performance metrics.



### Model evaluation and assessment


Model evaluation and assessment in machine learning refers to the process of measuring and analyzing a model's performance to determine its effectiveness in solving a specific task. It involves using metrics and techniques to quantify how well the model generalizes to unseen data, identifies patterns, and meets desired objectives, typically using a test dataset separate from the training data.

Below are common evaluation metrics by task types:

| Task types | Evaluation metrics |
| :--------: | :----------------: |
| Classification | Accuracy, precision, recall, F1-score, ROC-AUC, *etc.* |
| Regression | Mean Squared Error (MSE), Mean Absolute Error (MAE), <br>Root Mean Squared Error (RMSE), R-squared, *etc.* |
| Clustering | Silhouette score, Davies-Bouldin index, Calinski-Harabasz index |
| Ranking | Mean Reciprocal Rank (MRR), <br>Normalized Discounted Cumulative Gain (NDCG) |
| NLP or generative tasks | BLEU, ROUGE, perplexity (often overlaps with deep learning) |


Here are representative techniques and processes for the assessment:
- **train-validation-test split**: Divide data into training (model learning), validation (hyperparameter tuning), and test (final evaluation) sets to prevent overfitting.
- **cross-validation**: Use k-fold cross-validation to assess model stability across multiple data subsets.
- **confusion matrix**: For classification, visualize true positives, false negatives, etc.
- **learning curves**: Plot training *vs.* validation performance to diagnose underfitting or overfitting.
- **comparison with baselines**: Comparing model performance against simple baselines (*e.g.*, random guessing, linear models) to ensure meaningful improvement.
- **robustness testing**: Evaluate performance under noisy, adversarial, or out-of-distribution data.
- **fairness and bias analysis**: Assess model predictions for fairness across groups (*e.g.*, demographics).



### Hyperparameter tuning


Hyperparameter tuning is the process of optimizing the settings (hyperparameters) of a model that are not learned during training but significantly affect its performance. These include parameters like learning rate, number of hidden layers, or batch size, which control the model's behavior and training process.

The goal of this process is to find the best combination of hyperparameters that maximizes performance metrics (*e.g.*, accuracy, precision) on a validation set. 



### Model deployment, monitoring, and improvement


Model deployment, monitoring, and improvement refer to the processes involved in taking a trained machine learning model from development to production, ensuring it performs effectively in real-world applications, and continuously enhancing its performance.
- **model deployment** indicates an integration of a trained model into a production environment (APIs or cloud platforms) where it can make predictions or decisions on new, unseen data.
- Once deployed, the model’s performance must be continuously tracked to ensure it remains accurate and reliable over time, which is termed as **model monitoring**.
- As the models degrade over time, so continuous improvement is necessary. **model improvement** involves updating or retraining the model to maintain or enhance its performance based on monitoring insights or new data.



## Machine Learning Libraries



### Scikit-learn


**Scikit-learn** is a widely-used, open-source Python library designed for **classical machine learning**, offering a variety of algorithms and tools for for tasks, such classification, regression, clustering, and dimensionality reduction. It supports supervised learning (*e.g.*, SVM (support vector machine), decision trees, random forests), unsupervised learning (*e.g.*, k-means, PCA (principal component analysis)), and semi-supervised learning, with robust tools for data preprocessing, model evaluation, and hyperparameter tuning via ``GridSearchCV``. Built on NumPy, SciPy, and Matplotlib, it is designed for ease of use, making it ideal for beginners and rapid prototyping. Scikit-Learn excels in handling small to medium-sized datasets and includes utilities for data preprocessing, model evaluation, hyperparameter tuning, and pipeline construction. However, it lacks support for DL and GPU acceleration, limiting its scalability for large datasets or complex neural network tasks.



### Keras


**Keras** is a high-level neural networks API that simplifies the process of building and training DL models. Originally an independent library, Keras is now tightly integrated with TensorFlow as its official high-level interface (but also usable standalone), offering an accessible way to experiment with DL without sacrificing performance. Keras provides user-friendly abstractions for layers, models, loss functions, and optimizers, allowing users for quick prototyping of neural networks for tasks like image classification, text generation, and time series forecasting with minimal code. Keras abstracts away much of the complexity of TensorFlow while retaining flexibility, making it ideal for beginners and those who need fast experimentation.



### TensorFlow


Developed by Google, **TensorFlow** is a powerful open-source library primarily for DL but versatile enough for a broad range of ML tasks. It provides a flexible ecosystem for building complex models, including neural networks for computer vision, natural language processing, and time series analysis. TensorFlow supports distributed computing across CPUs, GPUs, and TPUs, making it suitable for both research and production at scale. Its robust features, such as TensorBoard for visualization, TensorFlow Serving for model deployment, and TensorFlow Lite for mobile inference, make it a comprehensive framework for end-to-end machine learning development. TensorFlow’s high-level Keras API simplifies model building, while its low-level operations provide flexibility for advanced research. TensorFlow is well-suited for tasks like image recognition, natural language processing (NLP), and reinforcement learning, though its complexity can pose a steeper learning curve for beginners compared to alternatives like PyTorch. 



### PyTorch


Developed by Facebook’s AI Research Lab (FAIR), PyTorch is auser-friendly and open-source DL library that has gained significant popularity in academia and industry. Known for its intuitive design and "define-by-run" (eager execution) approach, PyTorch allows developers to build, train, and debug models in a flexible and interactive manner. Its strong support for GPU acceleration and extensive ecosystem-ranging from computer vision (TorchVision) to NLP (TorchText) and audio (TorchAudio) -- make it an excellent choice for cutting-edge DL research and production. Popular in academia and increasingly in industry, PyTorch excels in rapid prototyping and experimentation but is less optimized for production deployment compared to TensorFlow. Its active community and support for GPU acceleration make it a favorite for cutting-edge ML and DL research.



### XGBoost & LightGBM


**XGBoost** (Extreme Gradient Boosting) and **LightGBM** (Light Gradient Boosting Machine) are high-performance gradient boosting libraries that have become go-to solutions for structured data problems, such as tabular datasets. Both libraries implement optimized gradient boosting algorithms that deliver fast training speeds, high accuracy, and scalability to large datasets. XGBoost is known for its robustness and versatility, while LightGBM offers further speed and memory efficiency through histogram-based algorithms and leaf-wise growth strategies. These libraries have become essential tools for data scientists working with structured data, outperforming traditional models in many real-world scenarios.



### Hugging Face Transformers


**Hugging Face Transformers** is a cutting-edge library that provides access to state-of-the-art pre-trained models for NLP tasks and computer vision, including text classification, translation, summarization, and question answering. The library’s pre-trained models and tokenizers simplify NLP workflows by enabling rapid experimentation with large language models, and in addition, this library supports both TensorFlow and PyTorch backends, integrating with datasets via Hugging Face’s datasets library, and has a vibrant community contributing to its continuous development.



### FastAI


**FastAI** is a high-level DL library built on PyTorch, designed to make AI accessible to a wider audience by simplifying complex tasks. It provides high-level abstractions and best practices out-of-the-box, allowing users to train powerful models with minimal code and optimal defaults. FastAI is particularly well-known for its transfer learning capabilities, enabling quick adaptation of pre-trained models for tasks like image classification and text generation. With its focus on practical usage, education, and strong community support, FastAI is ideal for beginners and practitioners who want to quickly deploy models without deep theoretical expertise.



### JAX

JAX, developed by Google, combines NumPy-like syntax with automatic differentiation and GPU/TPU acceleration, making it ideal for high-performance ML research. It enables composable function transformations (gradients, JIT compilation) and scales efficiently across hardware. While not as high-level as TensorFlow or PyTorch, JAX is favored for cutting-edge numerical computing, physics simulations, and advanced neural network research where speed and flexibility are crucial.



These libraries cater to different needs: Scikit-learn for classical ML, TensorFlow and PyTorch for DL and scalability, Keras for simplicity, XGBoost for high-performance tabular data tasks, and Hugging Face for transformer-based applications. The choice of these libraries depends on the task, data type, scalability needs, user expertise, and whether the focus is research, prototyping, or production deployment.

A summary of best features and key strengths of these libraries are summarized below.

| Library | Best Feature | Key Strength |
| :-----: | :----------: | :----------: |
| Scikit-Learn | Simple and consistent API <br>for classical ML tasks  <br>(classification, regression, clustering) <br>and small/medium datasets | Seamless integration with NumPy/Pandas <br>and extensive documentation for ease-of-use with <br>wide algorithm support |
| PyTorch | Dynamic computation graph (define-by-run) <br>for flexible model building and debugging | Flexible, intuitive framework with strong adoption <br>for academic research in DL tasks |
| TensorFlow | Scalability with GPU/TPU acceleration <br>for complex deep learning models | Excellent ecosystem (Keras, TF Hub, TF-Agents) <br>for production-scale applications |
| Keras | High-level, user-friendly API <br>for rapid prototyping | Simplifies construction of DL models, making it beginner-friendly <br>and efficient with TensorFlow compatibility <br>for quick model development |
| XGBoost & <br>LightGBM | Optimized gradient boosting algorithms | Extremely effective for high-performance <br>supervised learning with tabular/structured data |
| Hugging Face <br>Transformers | Extensive pretrained transformer models <br>for easy fine-tuning | Community-driven ecosystem with user-friendly pipelines <br>for NLP and vision tasks |
| FastAI | Transfer learning made easy <br>for NLP & vision tasks | Fast prototyping with minimal code and strong performance <br>for applied deep learning |
| JAX | NumPy + autodiff + GPU/TPU acceleration | Cutting-edge numerical computing, works with PyTorch/TensorFlow <br>via interoperability libraries, but offers lower-level control |



:::{keypoints}
- 
:::

