import umap.umap_ as umap 
import numpy as np 
from typing import Dict, List, Optional, Tuple
from sklearn.mixture import GaussianMixture
import pandas as pd 
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_engine.langchain_engine import * 
from utils import * 
import warnings
warnings.filterwarnings('ignore') 
RANDOM_SEED = 42  

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

def embed_cluster_summarize_texts(texts, level):
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
    
    template = """
    여기 이화여자대학교 학칙 문서가 있습니다.

    이 문서는 학칙의 핵심 내용을 포함하며, 총칙, 부설기관, 학사 운영, 학생 활동 및 행정 절차 등 주요 항목을 다룹니다.

    제공된 문서의 자세한 요약을 제공하십시오.

    문서:
    {context}
    """
    llm = get_llm(temperature=0)
    chain = get_chain(llm, template) 
    summaries = [] 
    for i in all_clusters:
        df_cluster = expanded_df[expanded_df["cluster"] == i] 
        #print(f"[DEBUGGING] {len(df_cluster)}")
        formatted_txt = fmt_txt(df_cluster) 
        summaries.append(chain.invoke({"context": formatted_txt})) 
    
    df_summary = pd.DataFrame(
        {
            "summaries": summaries,
            "level": [level] * len(summaries),
            "cluster": list(all_clusters),
        }
    )

    return df_clusters, df_summary

def recursive_embed_cluster_summarize(texts, level, n_levels):
    """Recursively embed/clustering/summarize texts until it single cluster remains or given level"""
    results = {} 

    df_clusters, df_summary = embed_cluster_summarize_texts(texts, level)

    results[level] = (df_clusters, df_summary) 

    unique_clusters = df_summary["cluster"].nunique()
    if level < n_levels and unique_clusters > 1:
        new_texts = df_summary["summaries"].tolist()
        next_level_results = recursive_embed_cluster_summarize(
            new_texts, level + 1, n_levels
        )
        results.update(next_level_results)

    return results


def main():
    load_env()
    #splits = load_ewha("./data") 
    docs = load_docs("./data/") 
    #splits = split_docs(docs, 300, 100) # Token limits..
    splits = split_docs(docs, 1000, 100)
    docs_texts = [d.page_content for d in splits] 

    # Make Tree 
    leaf_texts = docs_texts # Set document text to leaf text 

    results = recursive_embed_cluster_summarize(
        leaf_texts, level=1, n_levels=3
    ) 
    
    all_texts = leaf_texts.copy() 
    for level in sorted(results.keys()):
        # Extract summarization from the level
        summaries = results[level][1]["summaries"].tolist()
        # Add summarization to all_texts
        all_texts.extend(summaries)
    DB_INDEX = "./db/RAPTOR_faiss_original"
    embeddings = get_embedding() 
    if not os.path.exists(DB_INDEX):
        vectorstore = FAISS.from_texts(texts=all_texts, embedding=embeddings)
        vectorstore.save_local(folder_path=DB_INDEX)
        retriever = vectorstore.as_retriever()

    else:
        retriever = get_faiss(splits, save_dir=DB_INDEX) 

    prompt = """
                Please provide most correct answer from the following context.
                If the answer is not present in the context, please write "The information is not present in the context."
                ---
                Question: {question}
                ---
                Context: {context}
            """ 
    
    llm = get_llm(temperature=0)
    data_root = "./data"
    chain = get_qa_chain(llm, retriever, prompt) 

    print("[INFO] Load test dataset...") 
    questions, answers = read_data(data_root, filename="final_30_samples.csv") 

    responses = get_responses(chain=chain, prompts=questions)
    acc = eval(questions, answers, responses, debug=True) 

if __name__ == "__main__":
    main() 
