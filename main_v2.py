'''
Last Updated: 24.11.08
Add Retriever and Grounded
'''
from dotenv import load_dotenv
import os
import re
import pandas as pd
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_community.vectorstores import FAISS 
from langchain_upstage import UpstageLayoutAnalysisLoader
from langchain_upstage import UpstageGroundednessCheck
from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
    )

def load_env():
    global ROOT, UPSTAGE_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT, LANGCHAIN_API_KEY, CHUNK_SIZE, CHUNK_OVERLAP 
    load_dotenv(verbose=True)
    ROOT = os.getenv('ROOT')
    UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
    LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    CHUNK_SIZE = int(os.getenv('CHUNK_SIZE'))
    CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP'))

def split_docs():
    layzer = UpstageLayoutAnalysisLoader(
        api_key=UPSTAGE_API_KEY,
        file_path=os.path.join(ROOT, 'ewha.pdf'), 
        output_type="text"
    )

    docs = layzer.load()  # or layzer.lazy_load()
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP, 
        language=Language.HTML
    )
    splits = text_splitter.split_documents(docs)
    print("Splits:", len(splits))
    return splits

def get_chain():
    llm = ChatUpstage(api_key = UPSTAGE_API_KEY)
    prompt_template = PromptTemplate.from_template(
        """
        Please provide most correct answer from the following context.
        If question seems to be unrelated to the context, please think step by step.
        Otherwise, please summary the information you referred to along with the reasons why.
        NOTE) You MUST answer like this format: "[ANSWER]: (A) convolutional networks"
        ---
        Question: {question}
        ---
        Context: {context}
        """
    )
    chain = prompt_template | llm
    return chain

def read_data():
    data = pd.read_csv(os.path.join(ROOT, "test_10.csv"))
    prompts = data['prompts']
    answers = data['answers']
    # returns two lists: prompts and answers
    return prompts, answers

def init_retriever(splits):
    doc_embedder = UpstageEmbeddings(model="solar-embedding-1-large-passage")
    db = FAISS.from_documents(splits, doc_embedder)
    return db

def retrieve(db, query):
    query_embedder = UpstageEmbeddings(model="solar-embedding-1-large-query")
    query_vector = query_embedder.embed_query(query)
    context = db.similarity_search_by_vector(query_vector, k=4)
    return context

def grounded_check(context, answer):
    groundedness_check = UpstageGroundednessCheck()
    request_input = {
        "context": context,
        "answer": answer,
    }
    response = groundedness_check.invoke(request_input)
    #print(response)
    return response == "grounded" # grounded, notGrounded, or notSure

def get_responses(db, chain):
    # read samples.csv file
    prompts, answers = read_data()
    responses = []
    for prompt in prompts:
        context = retrieve(db, prompt)
        response = chain.invoke({"question": prompt, "context": context})
        #print(context)
        if not grounded_check(context, response.content) or "[ANSWER]:" not in response.content: # mmlu
            response = chain.invoke({"question": prompt, "context": context})
        responses.append(response.content)
    return answers, responses

def extract_answer(response):
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

def eval(answers, responses):
    cnt = 0
    for answer, response in zip(answers, responses):
        print("-"*10)
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

def main():
    # Load and set .env
    load_env()

    # Load and split ewha.pdf
    splits = split_docs()

    # Get langchain using template
    chain = get_chain()

    # Initialize FAISS retriever with ewha.pdf
    db = init_retriever(splits)

    # Get model's response from given prompts
    answers, responses = get_responses(db, chain)
    
    # Evaluation
    eval(answers, responses)

if __name__=="__main__":
    main()