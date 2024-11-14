'''
Last Updated: 24.11.06
Given Plain ver
'''
from dotenv import load_dotenv
import os
import re
import pandas as pd
import yaml
from tqdm import tqdm 

from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_text_splitters import Language,RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS 
from langchain_core.runnables import RunnablePassthrough


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_env(env_path=None):
    global UPSTAGE_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT, LANGCHAIN_API_KEY
    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv(verbose=True)
    UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
    LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


def split_docs(data_root, chunk_size, chunk_overlap):
    print("[INFO] Loading documents...")#For debugging
    layzer = UpstageLayoutAnalysisLoader(
        api_key=UPSTAGE_API_KEY,
        file_path=os.path.join(data_root, 'ewha.pdf'), 
        output_type="text",
        split="page"
    )

    docs = layzer.load()  # or layzer.lazy_load()
    print("[INFO] Spliting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        language=Language.HTML
    )

    splits = text_splitter.split_documents(docs)
    print("[INFO] # of splits:", len(splits))
    return splits

def get_llm(temperature=0):
    llm = ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=temperature)
    return llm 

def get_chain(llm, prompt, retriever=None):
    # You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    prompt_template = PromptTemplate.from_template(prompt)
    if retriever is not None:
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm 
            )
    else:
        chain = prompt_template | llm
    return chain

def read_data(data_path, filename="test_samples.csv"):
    data = pd.read_csv(os.path.join(data_path, filename))
    prompts = data['prompts']
    answers = data['answers']
    # returns two lists: prompts and answers
    return prompts, answers

def get_responses(chain, prompts):
    # read samples.csv file
    responses = []
    for prompt in tqdm(prompts, desc="Processing questions"):
        response = chain.invoke(prompt) # chain.invoke({"question": prompt, "context": context})
        responses.append(response.content)
    return responses

def get_embedding():
    # returns upstage embedding
    print("[INFO] Loading embeddings...")
    return UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model = 'solar-embedding-1-large')

def get_faiss(splits):
    # returns retriever FAISS 
    embeddings = get_embedding()
    print("[INFO] Get retriever FAISS ...")
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    return retriever 






def extract_answer(response):
    # funcion to extract an answer from response
    """
    extracts the answer from the response using a regular expression.
    expected format: "[ANSWER]: (A) convolutional networks"

    if there are any answers formatted like the format, it returns None.
    """
    pattern = r"\[ANSWER\]:\s*\((A|B|C|D|E)\)"  # Regular expression to capture the answer letter and text
    match = re.search(pattern, response)

    if match:
        return match.group(1) # Extract the letter inside parentheses (e.g., A)
    else:
        return extract_again(response) 
    
def extract_again(response):
    pattern = r"\b[A-J]\b(?!.*\b[A-J]\b)"
    match = re.search(pattern, response)
    if match: return match.group(0)
    else: return None


def eval(questions, answers, responses):
    cnt = 0
    for question, answer, response in zip(questions, answers, responses):
        print("-"*10)
        print(f"{question}\n")
        generated_answer = extract_answer(response)
        print(response)
        # check
        if generated_answer:
            print(f"generated answer: {generated_answer}, answer: {answer}")
        else:
            print("extraction fail")

        if generated_answer == None:
            continue
        if generated_answer in answer:
            cnt += 1
    print(f"acc: {(cnt/len(answers))*100}%")
    print("All Done")