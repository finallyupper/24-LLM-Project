'''
Last Updated: 24.11.16
- parent-child vectorspace (Opt: summary, hypo-queries)
- student-teacher structure
'''
from dotenv import load_dotenv
from datasets import load_dataset
from argparse import ArgumentParser
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever

import warnings
warnings.filterwarnings('ignore') 

from utils import * 
from langchain_engine_with_bm25_download import * # 원본 이름: langchain_engine

def load_custom_dataset(dataset_name):
    """Load a custom dataset by name."""
    if dataset_name == "arc": # use arc dataset
        return load_arc()
    elif dataset_name == "other dataset": # new dataset
        # add new load_dataset()
        return load_other_dataset()
    else:
        raise ValueError(f"[ERROR] Unsupported dataset: {dataset_name}")

def main(
    use_grounded,
    vec_layer,
    vec_store,
    dataset_name="arc", # (default: ARC)
    retriever_type="faiss", # Retriever type: faiss or chroma or bm25
    use_ensemble=True
):
    # Load configs
    load_env(".env")
    config = load_yaml("config.yaml")
    data_root = config['data_root']
    chunk_size = config['chunk_size'] 
    chunk_overlap = config['chunk_overlap']
    ewha_faiss_path = config['ewha_faiss_path']
    arc_faiss_path = config['arc_faiss_path']
    summ_chroma_path = config['summ_chroma_path']
    summ_faiss_path = config['summ_faiss_path']
    pc_chroma_path = config['pc_chroma_path']
    pc_faiss_path = config['pc_faiss_path']
    top_k = config['top_k']

    # Load and Split documents
    splits = load_ewha(data_root) # parent splits
    if "summ" in vec_layer:
        retriever = (lambda x: get_summ_chroma if "chroma" in vec_store else get_summ_faiss)(vec_store)(
                                    splits=splits, 
                                    top_k=top_k
                                    )
    elif "pc" in vec_layer:
        retriever = (lambda x: get_pc_chroma if "chroma" in vec_store else get_pc_faiss)(vec_store)(
                                splits=splits, 
                                top_k=top_k, 
                                chunk_size=chunk_size, chunk_overlap=chunk_overlap
                                )
        
    # Load custom dataset retriever
    custom_data = load_custom_dataset(dataset_name)
    if retriever_type == "faiss":
        custom_retriever = get_faiss(custom_data, save_dir=f"./db/{dataset_name}_faiss", top_k=top_k)
    elif retriever_type == "chroma":
        custom_retriever = get_chroma(custom_data, save_dir=f"./db/{dataset_name}_chroma", top_k=top_k)
    elif retriever_type == "bm25":
        custom_retriever = get_chroma(custom_data, save_dir=f"./db/{dataset_name}_chroma", top_k=top_k)
    else:
        raise ValueError("[ERROR] Invalid retriever_type value")
    
    # Apply ensemble retriever
    if use_ensemble:
        print("[INFO] Creating ensemble retriever...")
        retriever = get_ensemble_retriever(retriever, custom_retriever, w1=0.7, w2=0.3)


    # Make prompt template
    prompt = """
                Please provide most correct answer from the following context.
                If question seems to be unrelated to the context, ignore the given context.
                Otherwise, please summary the information you referred to along with the reasons why.
                NOTE) You MUST answer like this format: "[ANSWER]: (A) convolutional networks"
                ---
                Question: {question}
                ---
                Context: {context}
            """

    # Make llm
    llm = get_llm(temperature=0)

    # Get langchain using template
    chain = get_chain(llm, prompt, retriever=retriever) # retriever=None | retriever

    # Get model's response from given prompts
    print("[INFO] Load test dataset...") 
    questions, answers = read_data(data_root, filename="final_30_samples.csv") 
    responses = get_pc_responses(retriever, chain, prompts=questions, use_grounded=use_grounded)
    acc = eval(questions, answers, responses, debug=False)

if __name__=="__main__":
    PARSER = ArgumentParser()
    PARSER.add_argument("--use_grounded", action='store_true')
    PARSER.add_argument("-l", '--vec_layer', choices=["summ", "pc"], default="summ")
    PARSER.add_argument("-v", '--vec_store', choices=["faiss", "chroma"], default="faiss")
    main(**vars(PARSER.parse_args()))
