from utils import * 
import warnings
from argparse import ArgumentParser
from datasets import load_dataset
from langchain_engine.langchain_engine import *
warnings.filterwarnings('ignore') 

def main(
    ewha_ret1: str,
    ewha_ret2: str,
    arc_ret: bool,
):
    # Load configs
    load_env(".env")
    config = load_yaml("config.yaml") 
    data_root = config['data_root']
    chunk_size = config['chunk_size'] 
    chunk_overlap = config['chunk_overlap']
    ewha_faiss_path = config['ewha_faiss_path']
    ewha_bm25_path = config['ewha_bm25_path'] 
    arc_faiss_path = config['arc_faiss_path']
    arc_bm25_path = config['arc_bm25_path']
    summ_chroma_path = config['summ_chroma_path']
    summ_faiss_path = config['summ_faiss_path']
    pc_chroma_path = config['pc_chroma_path']
    pc_faiss_path = config['pc_faiss_path']
    raptor_faiss_path = config['raptor_faiss_path']
    top_k = config['top_k']

    # Load and Split documents
    splits = [];arc_data = []
    ret_dict = {
        "faiss": [split_docs, get_faiss, ewha_faiss_path],
        "bm25": [split_docs, get_bm25, ewha_bm25_path],
        "pc_faiss": [load_ewha, get_pc_faiss, pc_faiss_path],
        "pc_chroma": [load_ewha, get_pc_chroma, pc_chroma_path],
        "summ_faiss": [load_ewha, get_summ_faiss, summ_faiss_path],
        "summ_chroma": [load_ewha, get_summ_chroma, summ_chroma_path],
        "rap_faiss": [split_docs, get_faiss, raptor_faiss_path],
    }

    # Make embeddings, db, and rertriever 
    splits = ret_dict.get(ewha_ret1)[0](data_root, chunk_size, chunk_overlap) 
    ewha_retriever1  = ret_dict.get(ewha_ret1)[1](splits, save_dir=ret_dict.get(ewha_ret1)[2], top_k=top_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if ewha_ret2 is not None:
        splits = ret_dict.get(ewha_ret2)[0](data_root, chunk_size, chunk_overlap) 
        ewha_retriever2  = ret_dict.get(ewha_ret2)[1](splits, save_dir=ret_dict.get(ewha_ret2)[2], top_k=top_k, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        ewha_retriever_ensemble = get_ensemble_retriever(ewha_retriever1, ewha_retriever2, [0.5, 0.5])

    if arc_ret:
        if not os.path.exists(arc_faiss_path):
            arc_data = load_arc() 
        arc_retriever_faiss = get_arc_faiss(arc_data, save_dir="./db/arc_faiss",top_k=top_k) 
        arc_data = load_arc() 
        arc_retriever_bm25 = get_bm25(arc_data, top_k=top_k) 
        arc_retriever_ensemble = get_ensemble_retriever(arc_retriever_faiss, arc_retriever_bm25, w1=0.5, w2=0.5) 
    
    # Make prompt template please write "The information is not present in the context." and 
    prompt = """
                Please provide most correct answer from the following context.
                If the answer or related information is not present in the context, 
                solve the question without depending on the given context. 
                Please summarize the information you referred to along with the reasons why.
                You should give clear answer. Also, You are smart and very good at mathematics.
                 NOTE) You MUST answer like following format at the end.
                ---

                ### Example of expected format: 
                [ANSWER]: (A) convolutional networks
    
                ---
                ###Question: 
                {question}
                ---
                ###Context: 
                {context}
            """


    # Make llm
    llm = get_llm(temperature=0)

    # 8. Get langchain using template
    if ewha_ret2 is not None:
        chain = get_qa_chain(llm, ewha_retriever_ensemble, prompt_template=prompt)
    else: chain = get_qa_chain(llm, ewha_retriever1, prompt_template=prompt)
    if arc_ret:
        arc_chain = get_qa_chain(llm, arc_retriever_ensemble, prompt_template=prompt) 
        chain = get_agent_executor(llm, chain, arc_chain)

    # Get model's response from given prompts
    print("[INFO] Load test dataset...") 
    questions, answers = read_data(data_root, filename="final_30_samples.csv") 

    responses = get_responses(chain=chain, prompts=questions)
    acc1 = eval(questions, answers, responses, debug=True) 

    # If You want to compare two chains, use the below code!
    """ 
    ewha_chain = get_qa_chain(llm, arc_retriever_faiss, prompt_template=prompt) 
    print("[INFO] Load test dataset...") 
    questions, answers = read_data(data_root, filename="final_30_samples.csv") 
    responses = get_responses(chain=ewha_chain, prompts=questions)
    acc2 = eval(questions, answers, responses) 
    print(f">> Accuracy Comparison: {acc1} | {acc2}")  
    """
    # If you want to test you agent, use the code below
    # WARNINGS: It eats lots of money $$!!
    """ 
    responses = get_agent_responses(agent=agent, prompts=questions) 
    acc1 = eval(questions, answers, responses, debug=False) 
    """

if __name__=="__main__":
    ewha_retrievers_type = ["faiss", "bm25", "rap_faiss", "pc_faiss", "pc_chroma", "summ_faiss", "summ_chroma"]

    PARSER = ArgumentParser()
    PARSER.add_argument("-e1", '--ewha_ret1', choices=ewha_retrievers_type, default=None)
    PARSER.add_argument("-e2", '--ewha_ret2', choices=ewha_retrievers_type, default=None)
    PARSER.add_argument('--arc_ret', action='store_true')
    main(**vars(PARSER.parse_args()))
