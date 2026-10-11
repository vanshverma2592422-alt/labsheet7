import os
import re
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage


# ============================================================
# PROJECT 1: CUSTOMER SEGMENTATION USING K-MEANS CLUSTERING
# ============================================================

def customer_segmentation_project():
    """Segment customers and compare clustering techniques."""

    # --------------------------------------------------------
    # STEP 1: Load Dataset
    # --------------------------------------------------------

    file_path = "Mall_Customers.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Place Mall_Customers.csv in the project folder."
        )

    customers = pd.read_csv(file_path)

    # Standardize column names for easier processing
    customers.columns = (
        customers.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Expected columns after normalization:
    # customer_id, gender, age, annual_income_k, spending_score_1_100
    # Column matching below also supports common alternative names.

    def find_column(possible_names):
        for column in possible_names:
            if column in customers.columns:
                return column
        return None

    age_col = find_column(["age"])
    income_col = find_column([
        "annual_income_k",
        "annual_income",
        "income"
    ])
    spending_col = find_column([
        "spending_score_1_100",
        "spending_score",
        "spending"
    ])

    if not all([age_col, income_col, spending_col]):
        raise ValueError(
            "Dataset must contain age, annual income, "
            "and spending score columns."
        )

    # --------------------------------------------------------
    # STEP 2: Data Cleaning
    # --------------------------------------------------------

    numeric_features = [
        age_col,
        income_col,
        spending_col
    ]

    for column in numeric_features:
        customers[column] = pd.to_numeric(
            customers[column],
            errors="coerce"
        )

    customers = customers.dropna(
        subset=numeric_features
    ).copy()

    customers = customers.drop_duplicates()

    # Remove invalid negative values
    customers = customers[
        (customers[age_col] >= 0)
        & (customers[income_col] >= 0)
        & (customers[spending_col] >= 0)
    ].copy()

    if len(customers) < 3:
        raise ValueError(
            "At least three valid customer records are required."
        )

    # --------------------------------------------------------
    # STEP 3: Select Features
    # --------------------------------------------------------

    X = customers[numeric_features].copy()

    # --------------------------------------------------------
    # STEP 4: Feature Scaling
    # --------------------------------------------------------

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --------------------------------------------------------
    # STEP 5: Find Optimal K Using the Elbow Method
    # --------------------------------------------------------

    maximum_k = min(10, len(customers) - 1)
    k_values = range(1, maximum_k + 1)
    inertias = []

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(X_scaled)
        inertias.append(model.inertia_)

    plt.figure(figsize=(8, 5))
    plt.plot(list(k_values), inertias, marker="o")
    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("Customer Segmentation: Elbow Method")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 6: Compare K-Means for Different Cluster Counts
    # --------------------------------------------------------

    candidate_ks = range(
        2,
        min(10, len(customers) - 1) + 1
    )

    kmeans_scores = []
    kmeans_inertias = []

    for k in candidate_ks:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_scaled)

        kmeans_scores.append(
            silhouette_score(X_scaled, labels)
        )

        kmeans_inertias.append(model.inertia_)

    cluster_comparison = pd.DataFrame({
        "K": list(candidate_ks),
        "Inertia": kmeans_inertias,
        "Silhouette Score": kmeans_scores
    })

    # Choose K with the best silhouette score
    optimal_k = int(
        cluster_comparison.loc[
            cluster_comparison["Silhouette Score"].idxmax(),
            "K"
        ]
    )

    # --------------------------------------------------------
    # STEP 7: Train the Final K-Means Model
    # --------------------------------------------------------

    kmeans_model = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    customers["KMeans_Cluster"] = (
        kmeans_model.fit_predict(X_scaled)
    )

    kmeans_silhouette = silhouette_score(
        X_scaled,
        customers["KMeans_Cluster"]
    )

    # --------------------------------------------------------
    # STEP 8: Visualize K-Means Clusters
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=customers,
        x=income_col,
        y=spending_col,
        hue="KMeans_Cluster",
        palette="viridis",
        s=80
    )

    plt.title("Customer Segmentation Using K-Means")
    plt.xlabel("Annual Income")
    plt.ylabel("Spending Score")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 9: Analyze Customer Groups
    # --------------------------------------------------------

    customer_profiles = customers.groupby(
        "KMeans_Cluster"
    )[numeric_features].mean()

    customer_counts = customers[
        "KMeans_Cluster"
    ].value_counts().sort_index()

    # --------------------------------------------------------
    # STEP 10: Apply PCA for Two-Dimensional Visualization
    # --------------------------------------------------------

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    pca_customers = pd.DataFrame({
        "PC1": X_pca[:, 0],
        "PC2": X_pca[:, 1],
        "Cluster": customers["KMeans_Cluster"].to_numpy()
    })

    plt.figure(figsize=(8, 5))

    sns.scatterplot(
        data=pca_customers,
        x="PC1",
        y="PC2",
        hue="Cluster",
        palette="viridis",
        s=80
    )

    plt.title("Customer Clusters Using PCA")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 11: Apply Hierarchical Clustering
    # --------------------------------------------------------

    hierarchical_model = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage="ward"
    )

    customers["Hierarchical_Cluster"] = (
        hierarchical_model.fit_predict(X_scaled)
    )

    hierarchical_silhouette = silhouette_score(
        X_scaled,
        customers["Hierarchical_Cluster"]
    )

    # --------------------------------------------------------
    # STEP 12: Create a Dendrogram
    # --------------------------------------------------------

    hierarchy_matrix = linkage(
        X_scaled,
        method="ward"
    )

    plt.figure(figsize=(12, 6))

    dendrogram(
        hierarchy_matrix,
        truncate_mode="lastp",
        p=30
    )

    plt.xlabel("Customer Groups")
    plt.ylabel("Distance")
    plt.title("Customer Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 13: Compare Clustering Algorithms
    # --------------------------------------------------------

    algorithm_comparison = pd.DataFrame({
        "Algorithm": [
            "K-Means",
            "Hierarchical Clustering"
        ],
        "Silhouette Score": [
            kmeans_silhouette,
            hierarchical_silhouette
        ]
    })

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=algorithm_comparison,
        x="Algorithm",
        y="Silhouette Score"
    )

    plt.title("Customer Clustering Algorithm Comparison")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 14: Save Dataset, Models and Reports
    # --------------------------------------------------------

    customers.to_csv(
        "customer_segments.csv",
        index=False
    )

    cluster_comparison.to_csv(
        "customer_k_comparison.csv",
        index=False
    )

    customer_profiles.to_csv(
        "customer_cluster_profiles.csv"
    )

    algorithm_comparison.to_csv(
        "customer_algorithm_comparison.csv",
        index=False
    )

    joblib.dump(
        {
            "scaler": scaler,
            "kmeans_model": kmeans_model,
            "pca_model": pca,
            "feature_columns": numeric_features
        },
        "customer_segmentation_model.pkl"
    )

    return customers, customer_profiles, algorithm_comparison


# ============================================================
# PROJECT 2: NEWS ARTICLE CLUSTERING SYSTEM
# ============================================================

def clean_text(text):
    """Clean and normalize text before vectorization."""

    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def news_clustering_project():
    """Group similar news articles using TF-IDF and clustering."""

    # --------------------------------------------------------
    # STEP 1: Load Dataset
    # --------------------------------------------------------

    file_path = "news_articles.csv"

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            "Place news_articles.csv in the project folder."
        )

    news = pd.read_csv(file_path)

    news.columns = (
        news.columns
        .str.strip()
        .str.lower()
        .str.replace(r"[^a-z0-9]+", "_", regex=True)
        .str.strip("_")
    )

    # Accept common names for the article text column
    possible_text_columns = [
        "text",
        "article",
        "content",
        "description",
        "news",
        "body"
    ]

    text_column = next(
        (
            column for column in possible_text_columns
            if column in news.columns
        ),
        None
    )

    if text_column is None:
        raise ValueError(
            "Dataset must contain a text, article, content, "
            "description, news, or body column."
        )

    # --------------------------------------------------------
    # STEP 2: Data Cleaning
    # --------------------------------------------------------

    news = news.dropna(
        subset=[text_column]
    ).copy()

    news[text_column] = news[text_column].apply(clean_text)

    news = news[
        news[text_column].str.len() > 0
    ].copy()

    news = news.drop_duplicates(
        subset=[text_column]
    ).reset_index(drop=True)

    if len(news) < 3:
        raise ValueError(
            "At least three non-empty news articles are required."
        )

    # --------------------------------------------------------
    # STEP 3: Convert Text into TF-IDF Features
    # --------------------------------------------------------

    tfidf_vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=5000,
        min_df=1,
        max_df=1.0,
        ngram_range=(1, 2)
    )

    X_tfidf = tfidf_vectorizer.fit_transform(
        news[text_column]
    )

    # --------------------------------------------------------
    # STEP 4: Determine Candidate Cluster Counts
    # --------------------------------------------------------

    maximum_k = min(10, len(news) - 1)

    candidate_ks = range(2, maximum_k + 1)

    tfidf_inertias = []
    tfidf_silhouette_scores = []

    for k in candidate_ks:
        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(X_tfidf)

        tfidf_inertias.append(model.inertia_)

        tfidf_silhouette_scores.append(
            silhouette_score(
                X_tfidf,
                labels,
                metric="cosine"
            )
        )

    cluster_evaluation = pd.DataFrame({
        "K": list(candidate_ks),
        "Inertia": tfidf_inertias,
        "Silhouette Score": tfidf_silhouette_scores
    })

    # Select K with the best silhouette score
    optimal_k = int(
        cluster_evaluation.loc[
            cluster_evaluation["Silhouette Score"].idxmax(),
            "K"
        ]
    )

    # --------------------------------------------------------
    # STEP 5: Plot the Elbow Curve
    # --------------------------------------------------------

    plt.figure(figsize=(8, 5))

    plt.plot(
        list(candidate_ks),
        tfidf_inertias,
        marker="o"
    )

    plt.xlabel("Number of Clusters")
    plt.ylabel("Inertia")
    plt.title("News Clustering: Elbow Method")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 6: Train Final K-Means Model
    # --------------------------------------------------------

    kmeans_model = KMeans(
        n_clusters=optimal_k,
        random_state=42,
        n_init=10
    )

    news["KMeans_Cluster"] = (
        kmeans_model.fit_predict(X_tfidf)
    )

    kmeans_silhouette = silhouette_score(
        X_tfidf,
        news["KMeans_Cluster"],
        metric="cosine"
    )

    # --------------------------------------------------------
    # STEP 7: Reduce TF-IDF Dimensions for Visualization
    # --------------------------------------------------------

    if X_tfidf.shape[1] >= 2:
        svd_model = TruncatedSVD(
            n_components=2,
            random_state=42
        )

        X_reduced = svd_model.fit_transform(X_tfidf)

    else:
        # A one-dimensional fallback for very small vocabularies
        X_reduced = np.column_stack([
            X_tfidf.toarray().ravel(),
            np.zeros(X_tfidf.shape[0])
        ])

        svd_model = None

    news["Component_1"] = X_reduced[:, 0]
    news["Component_2"] = X_reduced[:, 1]

    # --------------------------------------------------------
    # STEP 8: Visualize K-Means News Clusters
    # --------------------------------------------------------

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=news,
        x="Component_1",
        y="Component_2",
        hue="KMeans_Cluster",
        palette="tab10",
        s=80
    )

    plt.title("News Article Clustering Using TF-IDF and K-Means")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 9: Extract Important Terms for Each Cluster
    # --------------------------------------------------------

    feature_names = tfidf_vectorizer.get_feature_names_out()
    cluster_keywords = {}

    for cluster_id in sorted(
        news["KMeans_Cluster"].unique()
    ):
        cluster_indices = np.where(
            news["KMeans_Cluster"].to_numpy() == cluster_id
        )[0]

        mean_tfidf = np.asarray(
            X_tfidf[cluster_indices].mean(axis=0)
        ).ravel()

        top_indices = mean_tfidf.argsort()[-10:][::-1]

        cluster_keywords[int(cluster_id)] = (
            feature_names[top_indices].tolist()
        )

    # --------------------------------------------------------
    # STEP 10: Apply Hierarchical Clustering
    # --------------------------------------------------------

    # Use a low-dimensional representation to limit memory use.
    n_components = min(
        50,
        X_tfidf.shape[1] - 1,
        X_tfidf.shape[0] - 1
    )

    if n_components >= 2:
        hierarchy_svd = TruncatedSVD(
            n_components=n_components,
            random_state=42
        )

        X_hierarchical = hierarchy_svd.fit_transform(
            X_tfidf
        )

    else:
        X_hierarchical = X_tfidf.toarray()

    # Normalize the reduced features before hierarchical clustering
    hierarchy_scaler = StandardScaler()
    X_hierarchical_scaled = hierarchy_scaler.fit_transform(
        X_hierarchical
    )

    hierarchical_model = AgglomerativeClustering(
        n_clusters=optimal_k,
        linkage="ward"
    )

    news["Hierarchical_Cluster"] = (
        hierarchical_model.fit_predict(
            X_hierarchical_scaled
        )
    )

    hierarchical_silhouette = silhouette_score(
        X_hierarchical_scaled,
        news["Hierarchical_Cluster"]
    )

    # --------------------------------------------------------
    # STEP 11: Create a Dendrogram
    # --------------------------------------------------------

    if len(news) <= 1000:
        hierarchy_matrix = linkage(
            X_hierarchical_scaled,
            method="ward"
        )

        plt.figure(figsize=(12, 6))

        dendrogram(
            hierarchy_matrix,
            truncate_mode="lastp",
            p=30
        )

        plt.xlabel("Article Groups")
        plt.ylabel("Distance")
        plt.title("News Article Hierarchical Dendrogram")
        plt.tight_layout()
        plt.show()

    # --------------------------------------------------------
    # STEP 12: Compare Clustering Techniques
    # --------------------------------------------------------

    algorithm_comparison = pd.DataFrame({
        "Algorithm": [
            "K-Means with TF-IDF",
            "Hierarchical Clustering"
        ],
        "Silhouette Score": [
            kmeans_silhouette,
            hierarchical_silhouette
        ]
    })

    plt.figure(figsize=(8, 5))

    sns.barplot(
        data=algorithm_comparison,
        x="Algorithm",
        y="Silhouette Score"
    )

    plt.title("News Clustering Algorithm Comparison")
    plt.tight_layout()
    plt.show()

    # --------------------------------------------------------
    # STEP 13: Prepare Article Titles for Inspection
    # --------------------------------------------------------

    title_column = next(
        (
            column for column in ["title", "headline"]
            if column in news.columns
        ),
        None
    )

    if title_column is None:
        news["Article_Title"] = (
            news[text_column].str.slice(0, 100)
        )
        title_column = "Article_Title"

    # --------------------------------------------------------
    # STEP 14: Save Clusters, Metrics and Trained Model
    # --------------------------------------------------------

    news.to_csv(
        "clustered_news_articles.csv",
        index=False
    )

    cluster_evaluation.to_csv(
        "news_k_comparison.csv",
        index=False
    )

    algorithm_comparison.to_csv(
        "news_algorithm_comparison.csv",
        index=False
    )

    # Save a report containing important terms for each cluster
    keyword_rows = []

    for cluster_id, keywords in cluster_keywords.items():
        keyword_rows.append({
            "Cluster": cluster_id,
            "Top_Terms": ", ".join(keywords)
        })

    pd.DataFrame(keyword_rows).to_csv(
        "news_cluster_keywords.csv",
        index=False
    )

    joblib.dump(
        {
            "tfidf_vectorizer": tfidf_vectorizer,
            "kmeans_model": kmeans_model,
            "svd_model": svd_model,
            "optimal_k": optimal_k,
            "text_column": text_column
        },
        "news_clustering_model.pkl"
    )

    return news, cluster_keywords, algorithm_comparison


# ============================================================
# PREDICT A CLUSTER FOR A NEW NEWS ARTICLE
# ============================================================

def predict_news_cluster(model_filename, article_text):
    """Assign a new article to a trained K-Means cluster."""

    saved_model = joblib.load(model_filename)

    vectorizer = saved_model["tfidf_vectorizer"]
    model = saved_model["kmeans_model"]

    cleaned_article = clean_text(article_text)

    article_vector = vectorizer.transform([
        cleaned_article
    ])

    cluster = model.predict(article_vector)

    return int(cluster[0])


# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    # Run Project 1 after placing Mall_Customers.csv
    # in the current project folder.
    #
    # customer_data, customer_profiles, customer_comparison = (
    #     customer_segmentation_project()
    # )

    # Run Project 2 after placing news_articles.csv
    # in the current project folder.
    #
    # news_data, news_keywords, news_comparison = (
    #     news_clustering_project()
    # )

    # Predict the cluster for a new news article after
    # training and saving the news clustering model.
    #
    # article_cluster = predict_news_cluster(
    #     "news_clustering_model.pkl",
    #     "Scientists announce a new medical research discovery."
    # )

    pass
