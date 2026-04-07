# DBLP Data Analysis - Trends and Research Paper Recommendations

This project focuses on analyzing one of the largest available datasets of Computer Science publications - the **DBLP** database. The main goal is to extract meaningful insights, analyze the evolution of research trends over the past decades, and implement a smart recommendation system powered by Artificial Intelligence.

## Instalation
* Please, before running code place DBLP dataset in "data" folder.
* To install frameworks use: 
* pip install -r requirements.txt

## Project Structure
Raport link for Notion: 
https://www.notion.so/DBLP-analysis-raport-33bf435493a480adbd23e1b20bb55343?showMoveTo=true&saveParent=true

The project is divided into core analytical modules contained within Jupyter Notebooks:

### 1. Topics and Trends Discovery (`topics_and_trends_discovery.ipynb`)
This module explores how research topics have evolved across three decades: **2004, 2014, and 2024**. Since the data lacks predefined labels, unsupervised machine learning methods were utilized:
* **Preprocessing:** Normalizing and cleaning XML data using the `lxml` library and NLP tools (e.g., stop words removal).
* **Embeddings:** Converting paper titles into vector representations using the `multilingual-e5-small` model.
* **Dimensionality Reduction:** Applying the **UMAP** (Uniform Manifold Approximation and Projection) algorithm for better visualization and to support clustering.
* **Clustering:** Using the **HDBSCAN** algorithm, which groups data based on density, automatically detecting thematic clusters.

### 2. Smart Research Papers Recommendation (`smart_research_papers_recommendation.ipynb`)
This module focuses on building a modern recommendation system using the **RAG** (Retrieval-Augmented Generation) architecture.
* The project compares two approaches to RAG: using the popular **LangChain** framework versus a custom "from scratch" implementation.
* It includes integration with vector databases (e.g., Chroma) to perform semantic searches for contextually related documents.

### 3. Helper Module (`src/data_analysis_helper.py`)
* This script contains the `DataAnalysisHelper` class, which provides utility functions for the main notebooks.
* It handles parsing large `.xml` files, cleaning text with regular expressions, and performing tokenization and stop word removal (using the NLTK library).

## Technologies Used
* **Language:** Python
* **Data Analysis:** Pandas, Numpy
* **NLP & Machine Learning:** NLTK, Sentence-Transformers, HDBSCAN, UMAP, BERTopic, Scikit-learn
* **Generative AI & RAG:** OpenAI API, LangChain, Chroma
* **Parsing:** lxml

## Graphs and Analysis
Graphic files located in the `graphs/` and `png/` folders contain charts showing initial dataset statistics (dataframes) and generated 2D projections (UMAP) of the discovered clusters.
