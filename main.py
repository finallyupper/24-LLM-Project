from utils import * 
import warnings
from datasets import load_dataset
from langchain_engine.langchain_engine import *
warnings.filterwarnings('ignore') 

def main():
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
    top_k = config['top_k']
    raptor_faiss_path = config['raptor_faiss_path']
    # Load and Split documents
    #ewha_faiss_path = "./db/ewha_faiss_fix"
    splits = [];arc_data = []
    if not os.path.exists(ewha_faiss_path):
        docs = load_docs(data_root) 
        splits = split_docs(docs, chunk_size, chunk_overlap) 
    if not os.path.exists(arc_faiss_path):
        arc_data = load_arc() 
    # Make embeddings, db, and rertriever 

    ewha_retriever_raptor_faiss = get_faiss(splits, save_dir=raptor_faiss_path, top_k=top_k) 
    #ewha_retriever_faiss = get_faiss(splits, save_dir=ewha_faiss_path, top_k=top_k) 
    ewha_retriever_bm25 = get_bm25(splits, ewha_bm25_path, top_k = top_k)
    ewha_retriever_ensemble = get_ensemble_retriever([ewha_retriever_raptor_faiss, ewha_retriever_bm25], [0.5, 0.5]) 

    arc_retriever_faiss = get_arc_faiss(arc_data, save_dir=arc_faiss_path, top_k=top_k) 
    arc_retriever_bm25 = get_bm25(arc_data, save_dir=arc_bm25_path, top_k=top_k) 
    arc_retriever_ensemble = get_ensemble_retriever([arc_retriever_faiss, arc_retriever_bm25], [0.5, 0.5])
    

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
    #total_ensemble = get_ensemble_retriever([ewha_retriever_raptor_faiss, ewha_retriever_ensemble, arc_retriever_ensemble], 
    #                                        [0.3, 0.4, 0.3])
    total_ensemble = get_ensemble_retriever([ewha_retriever_ensemble, arc_retriever_ensemble], 
                                            [0.5, 0.5])
    chain = get_qa_chain(llm, total_ensemble, prompt_template = prompt)
    
    #ewha_chain = get_qa_chain(llm, ewha_retriever_ensemble, prompt_template=prompt) 
    #arc_chain = get_qa_chain(llm, arc_retriever_ensemble, prompt_template=prompt) 
    #agent = get_agent_executor(llm, ewha_chain, arc_chain) 

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
    main()
