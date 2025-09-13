# Introduction to Machine Learning



:::{objectives}
- Provide a general overview of ML.
- Explain the relationship between AI, ML, and DL.
- Explore representative real-world applications of ML.
:::



:::{instructor-note}
- 15 min teaching
- 0 min exercising
:::



## What is Machine Learning


Machine learning (ML) is a field of computer science that studies algorithms and techniques for automating solutions to complex problems that are hard to program using conventional programing methods.

In conventional programming, the programmer explicitly codes the logic (rules) to transform inputs (data) into outputs (answers), making it suitable for well-defined, rule-based tasks. In ML, the system learns the logic (rules) from data and answers, making it ideal for complex, pattern-based tasks where explicit rules are hard to define. The choice between them depends on the problem, data availability, and complexity.

:::{figure} ./images/1-classic-programming-vs-ML.jpg
:align: center
:width: 75%

*Classic programming vs. machine learning.* [Source](https://twimlai.com/resources/kubernetes-for-mlops/)
:::



## Relation with Artificial Intelligence and Deep Learning


Artificial Intelligence (AI) is the broadest field, encompassing any technique that enables computers to mimic human intelligence, such as reasoning, problem-solving, perception, and decision-making. AI includes a wide range of approaches, from rule-based systems (like expert systems) to modern data-driven methods. It aims to create systems that can perform tasks that typically require human intelligence, such as playing chess, recognizing images, or understanding language.

ML is a subset of AI that focuses on algorithms and models that learn patterns from data to make predictions or decisions without being explicitly programmed. ML is one of the primary ways to achieve AI. It enables systems to improve performance over time by learning from experience (data) rather than relying solely on hardcoded rules. ML includes various techniques like supervised learning (*e.g.*, regression, classification), unsupervised learning (*e.g.*, clustering, dimensionality reduction), and reinforcement learning.

Deep Learning (DL) is a specialized subset of ML that uses neural networks with many layers (hence "deep") to model complex patterns in large datasets. DL is a subset of ML, and it leverages artificial neural networks inspired by the human brain to tackle tasks like image recognition, speech processing, and natural language understanding. DL excels in handling unstructured data (*e.g.*, images, audio, text) and requires significant computational power and large datasets for training.

:::{figure} ./images/1-relationship-AI-ML-DL.png
:align: center
:width: 50%

The relationship between artificial intelligence, machine learning, and deep learning. [Source](https://carpentries-lab.github.io/deep-learning-intro/)
:::



## Why Machine Learning?


ML is transforming how we solve complex problems in the real world by enabling systems to learn directly from data, rather than relying on explicitly programmed rules. In many real-world scenarios, such as medical diagnosis, stock market prediction, or natural language processing, the relationships between inputs and outputs are too complex or dynamic to define manually. ML models can uncover hidden patterns and make accurate predictions or decisions, making them essential tools in fields like healthcare, finance, transportation, and cybersecurity.

Another crucial advantage of ML is its ability to adapt and improve over time as more data becomes available. Unlike traditional rule-based systems that require constant manual updates, ML models can retrain and adjust themselves to new data, trends, or anomalies, ensuring that the system stays relevant and effective. For example, in fraud detection, ML algorithms can evolve as fraud tactics change, providing a stronger defense compared to static rules that may become outdated. This adaptability makes ML particularly powerful in dynamic, real-time environments where traditional programming methods fall short.

In addition, ML empowers the automation of complex tasks that were previously dependent on human expertise and intuition. From voice recognition in virtual assistants to autonomous driving, ML algorithms can process vast amounts of unstructured data such as text, images, and audio, which are traditionally challenging for computers to handle. By enabling machines to "learn" from experience and improve their performance over time, ML not only enhances productivity but also opens new frontiers for innovation across industries, creating smarter systems that can make meaningful contributions to society.



## Machine Learning Applications



### Problems can be solve with ML


ML is used across a wide range of industries and real-world problems in healthcare, finance, natural language processing, computer vision, transportation, manufacturing industry, retail, and cybersecurity.

Below are key categories of problems that can be applied using ML.

| Application area | Example use Cases |
| :--------------: | :---------------: |
| Healthcare | Disease prediction & diagnosis, <br>medical image analysis, drug discovery |
| Finance | Fraud detection, credit scoring, algorithmic trading
| Retail & e-commerce | Product recommendations, customer segmentation, <br>demand forecasting |
| Transportation & autonomous systems | Self-driving cars, traffic prediction, route optimization |
| Natural language processing (NLP) | Chatbots and virtual assistants, sentiment analysis, <br>language translation |
| Manufacturing & industry | Predictive maintenance, quality control, <br>supply chain optimization |
| Computer Vision | Facial recognition, object detection, image classification |



### Problems can't be solve with ML


ML is powerful, but it’s not magic. It’s a tool for finding patterns in data but has no idea what the patterns mean. Therefore it is not a substitute for human reasoning, creativity, or ethical judgment.

Below are key categories of problems that cannot be solved with ML due to inherent limitations, regardless of data or computational advancements.
- Problems with insufficient or poor-quality data: ML relies heavily on data. If data is scarce, noisy, biased, or unrepresentative, models fail to generalize. For example, predicting rare events with limited historical data (*e.g.*, catastrophic asteroid impacts, spread of pandemic) is unreliable.
- Problems requiring reasoning, understanding, or deep logic. ML models approximate patterns but don't understand them. They lack reasoning and common sense unless explicitly designed (*e.g.*, symbolic AI).
- Problems that involve subjective judgments or value-based decisions. ML models don't "know" what's right or wrong -- they reflect patterns in the data, including biases.
- Problems outside of distribution generalization. A model trained on photos of cats can't accurately classify dogs if it never saw dogs. ML models interpolate between known data. They struggle with novel scenarios far outside the training set.



### Problems can be, but shouldn't be solved with ML


There are many problems where ML (or DL) could technically be applied, but shouldn’t be -- either because of the simplicity of the problem or due to ethical, practical, or societal concerns.
- Tasks for modelling well defined systems, where the equations governing them are known and understood.
- Problems at high-stakes domains with unacceptable error rates: ML can predict outcomes in fields like medical diagnosis or aviation safety, but even small errors can lead to catastrophic consequences. Over-reliance on ML without human oversight risks lives when models fail in edge cases.
- Privacy-sensitive applications: ML can analyze personal data (*e.g.*, health records, browsing habits) to predict behaviors, but using it for invasive profiling, surveillance, or targeted manipulation (*e.g.*, hyper-personalized propaganda) raises serious privacy and autonomy concerns.
- Reinforcing harmful social norms: ML can optimize systems like targeted advertising or content recommendation, but doing so can amplify harmful behaviors (*e.g.*, echo chambers, misinformation, or addiction to social media) if not carefully regulated.

