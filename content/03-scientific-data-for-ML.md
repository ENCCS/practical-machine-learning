# Scientific Data for Machine Learning



:::{objectives}
- Gain an overview of different formats for scientific data.
- Understand common performance pitfalls when working with big data.
- Describe representative data storage formats and their pros and cons.
- Understand data structures for machine learning.
:::



:::{instructor-note}
- 30 min teaching
- 20 min exercising
:::



## Big Data


:::{discussion}
- How large is the data you are working with?
- Are you experiencing performance bottlenecks when you try to analyse it?
:::

Big Data refers to datasets that are so large, complex, or fast-changing that traditional data processing tools cannot handle them efficiently. It encompasses not only the sheer volume of data but also its variety, velocity, and veracity --- often summarized as the **4 Vs** of big data. These datasets can come from numerous sources, including social media, sensor networks, scientific experiments, and transactional systems. The ability to collect and analyze such massive amounts of information allows organizations and researchers to uncover trends, correlations, and insights that would be impossible to detect with smaller datasets.

The emergence of big data has transformed multiple domains, from business analytics and healthcare to climate science and genomics. Advanced computational methods, distributed storage systems, and parallel processing frameworks such as Hadoop and Spark have become essential for managing and analyzing these vast datasets. Efficient handling of big data enables organizations to make data-driven decisions, optimize operations, and identify opportunities for innovation.

In the context of ML, big data provides the raw material that fuels the learning process. Large and diverse datasets allow ML models to capture complex patterns, generalize well to unseen data, and improve predictive performance. Without sufficient and high-quality data, even the most sophisticated algorithms cannot perform effectively. From image recognition to natural language processing, every ML application depends on properly curated datasets for training, validation, and testing.
Therefore, data is the backbone of ML as it serves as the foundation for training models to recognize patterns, make predictions, and generate insights. In addition, data determines the applicability and scalability of ML solutions across domains, from scientific research to real-world applications.

In this episode, we will dive into the world of scientific data, examining the various types and forms it can take and understanding how it is organized and stored. We will explore different data storage formats, highlighting representative formats along with their respective advantages and limitations. This will provide a solid foundation for making informed decisions about how to handle and manipulate data effectively.
More importantly, we will focus on the data structures that are commonly used in ML and DL projects. Understanding these structures is essential for efficiently preparing, processing, and feeding data into models, ultimately enabling accurate predictions and insights. By the end of this session, you will have a clear understanding of how scientific data is organized and how it can be structured to support ML and DL workflows.



## Understanding Scientific Data


Scientific data refers to any form of data that is collected, observed, measured, or generated as part of scientific research or experimentation. This data is used to support scientific analysis, develop theories, and validate hypotheses. It can come from a wide range of sources, including experiments, simulations, observations, or surveys across various scientific fields.

In general, scientific data can be described ty two terms: **types of data** and **forms of data**. They are related but distinct --- types describe the nature of the data, while forms describe the how the data is structured and formatted (and stored, which will be discussed below).



### Types of scientific data


Types of scientific data refer to what the data represents. It focuses on the nature or category of the data content.
- **Bit and byte**: The smallest unit of storage in a computer is a **bit**, which holds either a 0 or a 1. Typically, eight bits are grouped together to form a **byte**. A single byte (8 bits) can represent up to 256 distinct values. By organizing bytes in various ways, computers can interpret and store different types of data.
- **Numerical data**: Different numerical data types (*e.g.*, integer and floating-point numbers) require different binary representation. Using more bytes for each value increases the range or precision, but it consumes more memory.
	- For example, integers stored with 1 byte (8 bits) have a range from [-128, 127], while with 2 bytes (16 bits) the range becomes [-32768, 32767]. Integers are whole numbers and can be represented exactly given enough bytes.
	- In contrast, floating-point numbers (used for decimals) often suffer from representation errors, since most fractional values cannot be precisely expressed in binary. These errors can accumulate during arithmetic operations. Therefore, in scientific computing, numerical algorithms must be carefully designed to minimize error accumulation. To ensure stability, floating-point numbers are typically allocated 8 bytes (64 bits), keeping approximation errors small enough to avoid unreliable results.
	- In ML/DL, half, single, and double precision refer to different formats for representing floating-point numbers, typically using 16, 32, and 64 bits, respectively.
		- **Single precision** (32-bit) is commonly used as a balance between computational efficiency and numerical accuracy.
		- **Half precision** (16-bit) offers faster computation and reduced memory usage, making it popular for training large models on GPUs, though it may suffer from lower numerical stability.
		- **Double precision** (64-bit) provides higher accuracy but is slower and more memory-intensive, so it's mainly used when high numerical precision is critical.
		- Many modern frameworks, like TensorFlow and PyTorch, support mixed precision training, combining half and single precision to optimize performance while maintaining stability.
- **Text data**: When it comes to text data, the simplest character encoding is ASCII (American Standard Code for Information Interchange), which was the most widely used encoding until 2008 when UTF-8 took over. The original ASCII uses only 7 bits for representing each character and therefore can encode 128 specified characters. Later, it became common to use an 8-bit byte to store each character, resulting in extended ASCII with support for up to 256 characters. As computers became more powerful and the need for including more characters from other alphabets, UTF-8 became the most common encoding. UTF-8 uses a minimum of one byte and up to four bytes per character. This flexibility makes UTF-8 ideal for modern applications requiring global character support.
- **Metadata**: Metadata encompasses diverse information about data, including units, timestamps, identifiers, and other descriptive attributes. While most scientific data is either numerical or textual, the associated metadata is usually domain-specific, and different types of data may have different metadata conventions. In scientific applications, such as simulations and experimental results, metadata is typically integrated with the corresponding dataset to ensure proper interpretation and reproducibility.



### Forms of scientific data


Forms of scientific data refer to how the data is structured or formatted. It focuses on the presentation or shape of the data.
- **Tabular data structure** (numerical arrays) is a collection of numbers arranged in a specific structure that one can perform mathematical operations on. Examples of numerical arrays are scalar (0D), row or column vector (1D), matrix (2D), and tensor (3D), *etc.*
- **Textual data structure** is a format for storing and organizing text-based data. It represents unstructured or semi-structured information as sequences of characters (letters, numbers, symbols, punctuation) arranged in strings.
- **Images, videos, and audio** are forms of scientific data that represent information through visual and auditory formats. Images capture static visual information as pixel arrays, videos combine sequential frames to show temporal changes, and audio encodes sound signals as time-series data for analysis.
- **Graphs and networks** are forms of scientific data that represent relationships between entities as nodes and connections as edges. They are used to model complex systems such as social networks, molecular interactions, and ecological food webs, capturing the structure and connectivity of scientific phenomena.



## Data Storage Format



### Representative data storage format


When it comes to data storage, there are many types of storage formats used in scientific computing and data analysis. There isn’t one data storage format that works in all cases, so choose a file format that best suits your data.

For tabular data, each column usually has a name and a specific data type while each row is a distinct sample which provides data according to each column (including missing values). The simplest way to save tabular data is using the so-called CSV (comma-separated values) file, which is human-readable and easily shareable. However, it is not the best format to use when working with big (numerical) data.

Gridded data is another very common data type in which numerical data is normally saved in a multi-dimensional grid (array). Common field-agnostic array formats include:
- **Hierarchical Data Format** (HDF5) is a high performance storage format for storing large amounts of data in multiple datasets in a single file. It is especially popular in fields where you need to store big multidimensional arrays such as physical sciences.
- **Network Common Data Form version 4** (NetCDF4) is a data format built on top of HDF5, but exposes a simpler API with a more standardised structure. NetCDF4 is one of the most used formats for storing large data from big simulations in physical sciences.
- **Zarr** is a data storage format designed for efficiently storing large, multi-dimensional arrays in a way that supports scalability, chunking, compression, and cloud-readiness.
- There are more file formats like [feather](https://arrow.apache.org/docs/python/feather.html), [parquet](https://arrow.apache.org/docs/python/parquet.html), [xarray](https://docs.xarray.dev/en/stable/) and [npy]( https://numpy.org/doc/stable/reference/routines.io.html) to store arrow tables or data frames.



### Overview of data storage format


Below is an overview of common data formats (✅ for *good*, 🟨 for *ok/depends on a case*, and ❌ for *bad*) adapted from Aalto university's [Python for scientific computing](https://aaltoscicomp.github.io/python-for-scicomp/work-with-data/#what-is-a-data-format).

| Name | Human <br>readable | Space <br>efficiency | Arbitrary <br>data | Tidy <br>data | Array <br>data | Long term <br>storage/sharing |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| Pickle        | ❌ | 🟨 | ✅ | 🟨 | 🟨 | ❌ |
| CSV           | ✅ | ❌ | ❌ | ✅ | 🟨 | ✅ |
| Feather       | ❌ | ✅ | ❌ | ✅ | ❌ | ❌ |
| Parquet       | ❌ | ✅ | 🟨 | ✅ | 🟨 | ✅ |
| npy           | ❌ | 🟨 | ❌ | ❌ | ✅ | ❌ |
| HDF5          | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| NetCDF4       | ❌ | ✅ | ❌ | ❌ | ✅ | ✅ |
| JSON          | ✅ | ❌ | 🟨 | ❌ | ❌ | ✅ |
| Excel         | ❌ | ❌ | ❌ | 🟨 | ❌ | 🟨 |
| Graph formats | 🟨 | 🟨 | ❌ | ❌ | ❌ | ✅ |



## Data Structures for ML/DL


ML (and DL) models require numerical input, so we must collect adaquate numerical data before training.
For ML tasks, multimedia data like image, audio, or video formats should be converted into tabular data or numerical arrays that ML models can process.
This conversion enables models to extract meaningful features, such as pixel intensities, audio frequencies or motion patterns, for tasks like classification or prediction.



### Numerical array 


Numerical array is a collection of numbers arranged in a specific structure that one can perform mathematical operations on. Examples of numerical arrays are scalar (0D), row or column vector (1D), matrix (2D), and tensor (3D), *etc.*

Python offers powerful libraries like NumPy, PyTorch, TensorFlow, and Dask (parallel Numpy) to work with numerical arrays (0D to *n*D).

```python
import numpy as np

# 0D (Scalar)
scalar = np.array(5)  

# 1D (Vector)
vector = np.array([1, 2, 3])  

# 2D (Matrix)
matrix_2D = np.array([[1, 2], [3, 4]])  

# 3D (Matrix)
matrix_3D = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
print(matrix_3D.shape)
```



### Tensor


In ML and DL, a tensor is a mathematical object used to represent and manipulate multidimensional data. It generalizes scalars, vectors, and matrices to higher dimensions, serving as the fundamental data structure in frameworks like TensorFlow and PyTorch.

Why to use tensors in ML/DL (advantages of Tensor)?
- Generalization of scalars/vectors/matrices: Tensors extend these concepts to any number of dimensions, which is essential for handling complex data like images (3D) and videos (4D+).
- Consistency: Tensors unify data structures across ML/DL frameworks, simplifying model building, training, and deployment.
- Efficient computation: Frameworks like TensorFlow and PyTorch optimize tensor operations for speed (using GPUs/TPUs).
- Neural network representations: Input data (images, text) is converted to tensors.
- Automatic differentiation: Tensors support gradient tracking, which is vital for backpropagation in neural networks.


:::{exercise} Tensor Creation and Operations

[HERE](./jupyter-notebooks/3-Tensor.ipynb) we provide a tutorial about Tensor including
- Tensor creation
- Tensor's properties (`shape`, `dtype`, `ndim`)
- Tensor operations
   - indexing, slicing, transposing
   - element-wise operations: addition, subtraction, *etc.*
   - matrix multiplication(`np.dot`, `torch.matmul`)
   - reshaping, flattening, squeezing, unsqueezing
   - reduction operations: sum, mean, max along axes
   - broadcasting: Rules and examples
- Tensors in DL frameworks
   - moving tensors between CPUs and GPUs (suppose that you can access to GPU cards)
:::


:::{keypoints}
- Key characteristics of big data --- volume, variety, velocity, and veracity. 
- High-quality, large datasets are essential for training effective machine learning models
- Scientific data has different types, including numerical, textual, and multimedia data, and these data can take different forms, such as tabular arrays, grids, and images.
- Scientific data are stored in various formats including CSV, HDF5, and others with their respective advantages and limitations.
- Tensors as a generalization of numerical data in machine learning (deep learning).
- Tensors allow models to efficiently handle complex, multidimensional data such as images, videos, and audio.
:::
