# 🧠 Modeling of Social Behaviour in Reddit Networks

This project focuses on **modeling social behavior and interaction patterns on Reddit** using large-scale hyperlink data between subreddits.  
The analysis combines **network science**, **community detection**, and **unsupervised clustering** to study activity, influence, sentiment, and structural organization in online social systems.

---

## 🎯 Research Goals

The project aims to:

- Model inter-subreddit interactions as directed weighted networks
- Analyze activity vs influence using network centrality measures
- Detect communities using structure-based methods (Louvain)
- Perform feature-based clustering using HDBSCAN
- Compare network communities vs semantic clusters
- Study concentration, fragmentation, and noise in social interaction data

---

## 📊 Data

The dataset is based on:

**Reddit Hyperlink Network**  
Subreddit-to-subreddit hyperlinks extracted from post titles and bodies  
(`SOURCE_SUBREDDIT → TARGET_SUBREDDIT`)

Each interaction includes:

- timestamp
- sentiment label (`+1`, `-1`, neutral)
- feature vector (`PROPERTIES`)

---

## 🔄 Data Preprocessing

Steps:

1. Merge title and body hyperlink datasets
2. Create a vocabulary mapping of subreddits → integer IDs
3. Convert the data into a numerical edge list
4. Save:
   - `preprocessed_data.csv`
   - `mapping.json`

All preprocessing utilities are implemented in `utils.py`.

---

## 🌐 Network Construction

Networks are modeled using **NetworkX**:

- Directed graph
- Weighted edges (interaction counts)
- Optional filtering by sentiment:
  - full network
  - positive-interaction subnetwork

Graphs are stored in **GraphML** format for reproducibility.

---

## 📈 Centrality Analysis

Implemented measures:

- Degree centrality
- Weighted in-degree (influence)
- Weighted out-degree (activity)
- Betweenness centrality
- Closeness centrality
- Eigenvector centrality

Additional analyses:

- Concentration of activity among top influential nodes

Results are exported as:

- CSV tables
- JSON rankings
- PNG visualizations

---

## 🧩 Community Detection (Louvain)

Community detection is performed on the **positive-interaction network** using the **Louvain algorithm**:

- Modularity optimization
- Identification of densely connected subgraphs

Output:

- full community compositions
- distribution of community sizes

Results are stored in `data/community/`.

---

## 🔍 Feature-Based Clustering (HDBSCAN)

Unsupervised clustering is applied to subreddit feature vectors.

Pipeline:

1. Parse feature vectors (`PROPERTIES`)
2. Standardize features
3. Reduce dimensionality using **PCA**
4. Cluster using **HDBSCAN**

Advantages:

- Automatic number of clusters
- Explicit noise detection
- Robust to non-spherical distributions

Outputs:

- cluster assignments
- cluster size distributions
- noise statistics

---

## ⚖️ Community vs Clustering Comparison

The project compares:

- Louvain communities (topology-based)
- HDBSCAN clusters (feature-based)

The comparison highlights differences between **structural** and **semantic** representations of social behavior.

---

## 🛠️ Technologies Used

- Python 3.10+
- NetworkX
- Pandas
- NumPy
- Scikit-learn
- Plotly
- HDBSCAN
