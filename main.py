from utils import * 
import warnings
warnings.filterwarnings('ignore') 

def main():
    # 0. Load configs
    load_env(".env")
    config = load_yaml("config.yaml") 
    data_root = config['data_root']
    chunk_size = config['chunk_size'] 
    chunk_overlap = config['chunk_overlap']
    print(f'[INFO] Chunk size: {chunk_size} | Chunk overlap: {chunk_overlap}')

    # 1, 2. Load and Split documents
    splits = split_docs(data_root, chunk_size, chunk_overlap) 

    # 3,4,5. make embeddings, db, and rertriever 
    retriever = get_faiss(splits) 

    # 6. make prompt template
    prompt = """
                Please provide most correct answer from the following context.
                If the answer is not present in the context, please write "The information is not present in the context."
                ---
                Question: {question}
                ---
                Context: {context}
            """
    # 7. make llm
    llm = get_llm(temperature=0)

    # 8. Get langchain using template
    chain = get_chain(llm, prompt, retriever)


    # 9. Get model's response from given prompts
    print("[INFO] Load test dataset...") 
    questions, answers = read_data(data_root,filename="test_samples.csv") 
    responses = get_responses(chain=chain, prompts=questions)
    
    # 10. Evaluation
    eval(questions, answers, responses)

if __name__=="__main__":
    main()