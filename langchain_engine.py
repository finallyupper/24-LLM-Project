from dotenv import load_dotenv
import os
import threading
import itertools
import time
from tqdm import tqdm 

from langchain_core.prompts import PromptTemplate
from langchain_upstage import ChatUpstage, UpstageEmbeddings, UpstageLayoutAnalysisLoader
from langchain_text_splitters import Language,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from datasets import load_dataset
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain import hub
from langchain.tools.retriever import create_retriever_tool

from utils import format_docs, format_arc_doc

def load_env(env_path=None):
    """Loads API keys"""
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


def load_docs(data_root):
    """Loads documentary using UpstageLayoutAnalysisLoader"""
    layzer = UpstageLayoutAnalysisLoader(
        api_key=UPSTAGE_API_KEY,
        file_path=os.path.join(data_root, 'ewha.pdf'), 
        output_type="text",
        split="page"
    )
    def loading_animation():
        for c in itertools.cycle(['|', '/', '-', '\\']):
            if not loading: 
                break
            print(f"\r[INFO] Loading documents... {c}", end="")
            time.sleep(0.1)
        print("\r[INFO] Loading documents... Done!    ")
    loading = True
    t = threading.Thread(target=loading_animation)
    t.start()

    docs = layzer.load()  # or layzer.lazy_load() 
    loading = False
    t.join()  
    return docs 

def split_docs(docs, chunk_size, chunk_overlap):
    """Returns splits of docs using given chunk size and overlap"""
    print("[INFO] Spliting documents...")
    text_splitter = RecursiveCharacterTextSplitter.from_language(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap, 
        language=Language.HTML
    )

    splits = text_splitter.split_documents(docs)
    print("[INFO] # of splits:", len(splits))
    return splits


def get_embedding():
    """Loads upstage embedding"""
    # returns upstage embedding
    print("[INFO] Loading embeddings...")
    return UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model = 'solar-embedding-1-large')


def get_llm(temperature=0):
    """Loads LLM model from Upstage"""
    llm = ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=temperature)
    return llm 

def get_qa_chain(llm, retriever, prompt_template=None):
    """
    Loads LLM chain. 
    If customed prompt template is not given, 
    this function will apply huggingface's hub QA prompt named rlm/rag-prompt as our prompt template.
    returns chain
    """
    if prompt_template is not None:
         prompt_template = PromptTemplate.from_template(prompt_template)
    else:
        prompt_template = hub.pull("rlm/rag-prompt") #QA prompt  https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=5b2073af-2123-4ed3-b218-fa406e467d84 
    
    rag_chain = (
        {"context": retriever | format_docs , "question": RunnablePassthrough()}
        | prompt_template
        | llm
    )

    return rag_chain 
    
def get_responses(chain, prompts):
    # read samples.csv file
    responses = []
    for prompt in tqdm(prompts, desc="Processing questions"):
        response = chain.invoke(prompt) # chain.invoke({"question": prompt, "context": context})
        responses.append(response.content)
    return responses

def get_agent_responses(agent, prompts):
    """Get responses from given agent"""
    responses = [] 
    for prompt in tqdm(prompts, desc="Processing questions"):
        response = agent.invoke({"input": prompt}) 
        responses.append(response)
    return responses 

        
def get_chroma(splits, save_dir="./db/chroma", top_k=4, collection_name=""):
    embeddings = get_embedding() 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) 
        vectorstore = Chroma.from_documents(collection_name=collection_name, 
                                            documents=splits, 
                                            embedding=embeddings, 
                                            persist_directory=save_dir) 
    else:
        # If the db already exists, load from local.
        vectorstore = Chroma(
            persist_directory=save_dir,
            embedding_function=embeddings,
            collection_name=collection_name
        )    
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever 


def get_faiss(splits, save_dir="./db/faiss", top_k=4): 
    # returns retriever FAISS 
    embeddings = get_embedding()
    print("[INFO] Get retriever FAISS ...")
    if not os.path.exists(save_dir):
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        os.mkdir(save_dir) 
        vectorstore.save_local(save_dir)
        print("[INFO] Successfully saved Vectorscore to local!")
    else:
        # If the db already exists, load from local.
        vectorstore = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True) 
        print(f"[INFO] Load DB from {save_dir}...") 

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k}) # default = 4
    return retriever 

def get_bm25(splits, top_k=4):
    bm25_retriever  = BM25Retriever.from_documents(documents=splits)
    bm25_retriever.k = top_k 
    return bm25_retriever 

def get_ensemble_retriever(r1, r2, w1=0.7, w2=0.3):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[r1, r2], weights=[w1, w2]
        )
    return ensemble_retriever 

def load_arc():
    """Loads allenai/ai2_arc dataset and make it as metadata"""
    ds = load_dataset("allenai/ai2_arc", "ARC-Easy")
    train_data = ds['train']
    train_docs = []
    for entry in train_data:
        doc_content = format_arc_doc(entry)
        doc = Document(page_content=doc_content, metadata={"question": entry['question'], "choices": entry['choices']})
        train_docs.append(doc)
    return train_docs 

def get_arc_faiss(arc_data, save_dir="./db/arc_faiss", top_k=4):
    """Get FAISS retriever from arc dataset"""
    retriever_arc = get_faiss(arc_data, save_dir, top_k) 
    return retriever_arc 

def get_arc_chroma(arc_data, save_dir="./db/arc_chroma", top_k=4, collection_name="chroma"):
    """Get Chroma retriever from arc dataset"""
    retriever_arc = get_chroma(arc_data, save_dir, top_k=top_k, collection_name=collection_name)   
    return retriever_arc
    # ex. arc_retriever = get_arc_chroma(arc_data, save_dir="./db/arc_chroma", top_k=top_k, collection_name="arc_chroma") 

def get_agent_executor(llm, r1, r2):
    """Make agent executor with given two retrievers, r1 and r2."""
    retriever_tool_1 = create_retriever_tool(r1, "ewha_search", "Searches any questions related to school rules. Always use this tool when user query is related to EWHA or school rules!") 
    retriever_tool_2 = create_retriever_tool(r2, "arc_search", "Searches any questions related to science. Always use this tool when user query is related to science!")
    tools = [retriever_tool_1, retriever_tool_2]
    
    prompt = hub.pull("hwchase17/openai-functions-agent")
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools)
    
    return agent_executor

# Old version.
def get_chain(llm, prompt, retriever=None):
    # You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    prompt_template = PromptTemplate.from_template(prompt)
    if retriever is not None:
        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm )
    else:
        chain = prompt_template | llm
    return chain
