import umap.umap_ as umap 
import numpy as np 
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
import pandas as pd 
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_engine.langchain_engine import * 
from utils import * 
import warnings
from datasets import load_dataset
from prompts import *
warnings.filterwarnings('ignore') 
RANDOM_SEED = 42  
# https://github.com/teddylee777/langchain-kr/blob/main/12-RAG/09-RAPTOR-Long-Context-RAG.ipynb 

def global_cluster_embeddings(embeddings, dim, n_neighbors=None, metric="cosine"):
    """Globally reduce dimension using UMAP"""
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

def local_cluster_embeddings(embeddings, dim, num_neighbors=10, metric="cosine"):
    """Locally reduce dimension for embeddings"""
    return umap.UMAP(n_neighbors=num_neighbors, n_components=dim, metric=metric).fit_transform(embeddings) 

def embed_func(texts):
    embeddings = get_embedding() 
    embedded_query = embeddings.embed_documents(texts) 
    embedded_query = np.array(embedded_query)
    return embedded_query

def get_optimal_clusters(embeddings, max_clusters=50, random_state=RANDOM_SEED):
    """Find optimal number of clusters using BIC through GMM"""
    max_clusters = min(max_clusters, len(embeddings)) 
    n_clusters = np.arange(1, max_clusters)
    bics = [] # list to save BIC(Bayesian Information Criterian) scores
    for n in n_clusters:
        gm = GaussianMixture(n_components=n, random_state=random_state)
        gm.fit(embeddings) 
        bics.append(gm.bic(embeddings))
    return n_clusters[np.argmin(bics)]  

def GMM_cluster(embeddings, threshold, random_state=0):
    n_clusters = get_optimal_clusters(embeddings=embeddings, random_state=random_state)  
    # Initialize GM model
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    # Train
    gm.fit(embeddings) 

    # Predict
    probs = gm.predict_proba(embeddings) 
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters   

def perform_clustering(embeddings, dim, threshold):
    """Performs Clustering"""
    if len(embeddings) <= dim + 1: # Not enough data
        return [np.array([0]) for _ in range(len(embeddings))] 
    # Global Dim reduction
    reduced_embeddings_global = global_cluster_embeddings(embeddings, dim) 
    # Global clustering
    global_clusters, n_global_clusters = GMM_cluster(reduced_embeddings_global, threshold)
    
    all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
    total_clusters = 0

    for i in range(n_global_clusters):
        # Get embeddings that are included in global cluster
        _global_cluster_embeddings = embeddings[np.array([i in gc for gc in global_clusters])]
        if len(_global_cluster_embeddings) == 0:
            continue
        if len(_global_cluster_embeddings) <= dim + 1:
            local_clusters = [np.array([0]) for _ in _global_cluster_embeddings]
            n_local_clusters = 1
        else:
            # Local dim reduction and clustering
            reduced_embeddings_local = local_cluster_embeddings(
                _global_cluster_embeddings,
                dim
            )
            local_clusters, n_local_clusters = GMM_cluster(
                reduced_embeddings_local, threshold
            )

        for j in range(n_local_clusters):
            _local_cluster_embeddings = _global_cluster_embeddings[
                np.array([j in lc for lc in local_clusters])
            ]
            indices = np.where(
                (embeddings == _local_cluster_embeddings[:, None]).all(-1)
            )[1]
            for idx in indices:
                all_local_clusters[idx] = np.append(
                    all_local_clusters[idx], j + total_clusters
                )
        total_clusters += n_local_clusters

    return all_local_clusters 

def embed_clusters_texts(texts):
    """
    Embeds texts and clustering. 
    Returns dataframe containing texts, embeddings, and cluster labels
    """
    text_embeddings_np = embed_func(texts) # CHECK 
    cluster_labels = perform_clustering(embeddings=text_embeddings_np,
                                        dim=10,
                                        threshold=0.1) 
    df = pd.DataFrame() 
    df["text"] = texts 
    df["embd"] = list(text_embeddings_np)
    df["cluster"] = cluster_labels
    return df 

def fmt_txt(df: pd.DataFrame) -> str: # CHECK if it is our best ...
    unique_txt = df["text"].tolist() 
    return "--- --- \n --- --- ".join(
        unique_txt
    )  

def choose_template(type):
    if type =="law":
        template = RAPTOR_LAW_TAMPLATE 
    elif type == "psychology":
        template = RAPTOR_PSYCHOLOGY_TEMPLATE
    elif type == "business":
        template = RAPTOR_BUSINESS_TEMPLATE
    elif type == "philosophy":
        template = RAPTOR_PHILOSOPHY_TEMPLATE 
    return template 

def embed_cluster_summarize_texts(texts, level, type):
    df_clusters = embed_clusters_texts(texts) 
    expanded_list = [] 
    
    for idx, row in df_clusters.iterrows():
        for cluster in row["cluster"]:
            expanded_list.append({"text": row["text"], 
                                  "embd": row["embd"], 
                                  "cluster": cluster})
    
    expanded_df = pd.DataFrame(expanded_list)
    all_clusters = expanded_df["cluster"].unique() 
    print(f"--Generated {len(all_clusters)} clusters--") 
    
    template = choose_template(type=type)  

    llm = get_llm(temperature=0)
    chain = get_chain(llm, template) 
    summaries = [] 
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i] 
        #print(f"[DEBUGGING] {len(df_cluster)}")
        formatted_txt = fmt_txt(df_cluster) 
        summaries.append(chain.invoke({"doc": formatted_txt})) 
    
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary

def recursive_embed_cluster_summarize(texts, level, n_levels, type):
    """Recursively embed/clustering/summarize texts until it single cluster remains or given level"""
    results = {} 

    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level, type)

    results[level] = (df_clusters, df_summary) 

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels, type
        )
        results.update(next_level_results)

    return results

def save_raptor(type, save_path, n_levels=3):
    load_env()
    splits = load_customed_datasets(type = type)

    chunk_size_tok = 1000; chunk_overlap=0
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tok, 
        chunk_overlap=chunk_overlap
    )
    print("[INFO] Splitting documents ...")
    texts_split = text_splitter.split_documents(splits)
    docs_texts = [d.page_content for d in texts_split] 

    # Make Tree 
    print("[INFO] Making tree ...")
    leaf_texts = docs_texts # Set document text to leaf text 
    results = recursive_embed_cluster_summarize(
        leaf_texts, level=1, n_levels=n_levels, type=type
    ) 

    all_texts = leaf_texts.copy() 
    print("[INFO] Adding summarization ...")
    for level in tqdm(sorted(results.keys())):
        # Extract summarization from the level
        summaries = results[level][1]["summaries"].tolist()
        # Add summarization to all_texts
        all_texts.extend(summaries)

    if not os.path.exists(save_path):
        vectorstore = FAISS.from_texts(texts=all_texts, embedding=get_embedding() )
        vectorstore.save_local(folder_path=save_path)
        print(f"[INFO] Saved Vector DB into {save_path}")

if __name__ == "__main__":
    # law, psychology, business, philosophy
    # PROCESSING ...
    # save_raptor(type="law", save_path="/home/yoojinoh/Others/NLP/24-LLM-Project/db/raptor/law", n_levels=4) 
    
    # TOO HUGE ... (-> 20k random sampling)
    save_raptor(type="philosophy", save_path="/home/yoojinoh/Others/NLP/24-LLM-Project/db/raptor/philosophy", n_levels=3)
    # save_raptor(type="psychology", save_path="/home/yoojinoh/Others/NLP/24-LLM-Project/db/raptor/psychology", n_levels=7) 

    # DONE 
    # save_raptor(type="business", save_path="/home/yoojinoh/Others/NLP/24-LLM-Project/db/raptor/business", n_levels=4)
    