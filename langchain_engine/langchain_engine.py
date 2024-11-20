from dotenv import load_dotenv
import os
import re
import pickle
import threading
import itertools
import time
import json
import uuid
from tqdm import tqdm 
import torch 
from collections import Counter
from sys import exit

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.stores import InMemoryByteStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers.openai_functions import JsonKeyOutputFunctionsParser
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_upstage import ChatUpstage, UpstageEmbeddings, UpstageLayoutAnalysisLoader, UpstageGroundednessCheck
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
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import format_docs, format_arc_doc

def load_env(env_path=None):
    """Loads API keys"""
    global ROOT, UPSTAGE_API_KEY, LANGCHAIN_TRACING_V2, LANGCHAIN_ENDPOINT, LANGCHAIN_PROJECT, LANGCHAIN_API_KEY
    if env_path is not None:
        load_dotenv(env_path)
    else:
        load_dotenv(verbose=True)
    ROOT = os.getenv('ROOT')
    UPSTAGE_API_KEY = os.getenv('UPSTAGE_API_KEY')
    LANGCHAIN_TRACING_V2 = os.getenv('LANGCHAIN_TRACING_V2')
    LANGCHAIN_ENDPOINT = os.getenv('LANGCHAIN_ENDPOINT')
    LANGCHAIN_PROJECT = os.getenv('LANGCHAIN_PROJECT')
    LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')
    #os.chdir(ROOT)

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

def split_docs(data_root="./data", chunk_size=300, chunk_overlap=100):
    docs = load_docs(data_root)

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

def get_faiss_vs(splits, embeddings):
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings) 
    return vectorstore 
       
def get_faiss(splits, save_dir="./db/ewha/ewha_faiss_fix", chunk_size=None, chunk_overlap=None, top_k=4): 
    # returns retriever FAISS 
    embeddings = get_embedding()
    print("[INFO] Get retriever FAISS ...")
    if not os.path.exists(save_dir):
        vectorstore = get_faiss_vs(splits, embeddings)
        os.mkdir(save_dir) 
        vectorstore.save_local(save_dir)
        print("[INFO] Successfully saved Vectorscore to local!")
    else:
        # If the db already exists, load from local.
        vectorstore = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True) 
        print(f"[INFO] Load DB from {save_dir}...") 

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k}) # default = 4

    return retriever 

def get_bm25(splits, save_dir="./db/bm25", chunk_size=None, chunk_overlap=None, top_k=4):
    # Where to save BM25
    bm25_path = os.path.join(save_dir, "bm25.pkl")

    # Load BM25 from local
    print("[INFO] Get retriever BM25 ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(bm25_path):
        print("[INFO] Creating BM25 index...")
        # Make BM25
        print(splits)
        bm25_retriever = BM25Retriever.from_documents(documents=splits)
        bm25_retriever.k = top_k
        # Save BM25
        with open(bm25_path, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print(f"[INFO] Successfully saved BM25 index to {bm25_path}!")
    else:
        # Load BM25 from local
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)
        print(f"[INFO] Load BM25 index from {bm25_path}...")

    bm25_retriever.k = top_k
    return bm25_retriever

def get_ensemble_retriever(retrievers, weights):
    ensemble_retriever = EnsembleRetriever(retrievers=retrievers, weights=weights)
    return ensemble_retriever  

def remove_header(text):
    return text.replace("이화여자대학교 학칙", "")

def to_document(text: str, meta):
    if meta:
        return Document(id=meta, page_content=remove_header(text), metadata={"p_id": meta})
    else: return Document(id=meta, page_content=text, metadata={"p_id": meta})
                
def load_ewha(data_root, chunk_size=None, chunk_overlap=None, json_name = "ewha_chunk_doc.json"):
    """
    Corrects the spacing in the given documents using chain and save as json file
    returns splits list
    """
    ewha_chunks_path = os.path.join(data_root, json_name)

    if not os.path.exists(ewha_chunks_path):
        filename = os.path.join(data_root, "ewha_full_text.txt")
        f = open(filename, 'r')
        text = f.read()

        pattern1 = r"제\d+장"
        pattern2 = r'\[별표 \d+\] '
        matches1 = re.findall(pattern1, text)
        splitted = re.split(pattern1, text)
        matches2 = re.findall(pattern2, splitted[-1]) 
        
        splits = [to_document(splitted[0], 0)] \
            + [to_document(p + chunk, i+1) for i, (p, chunk) in enumerate(zip(matches1, splitted[1:-1]))] \
            + [to_document(p + chunk, i+len(matches1)+1) for i, (p, chunk) in enumerate(zip(matches2, re.split(pattern2, splitted[-1])))]
        llm = get_llm(temperature=0) 

        chain = (
                {"doc": lambda x: x.page_content} # Use as a context
                | ChatPromptTemplate.from_template("Please correct the spacing in the given text. It is a university regulation document. Just give only answer.\n\n{doc}")
                | llm
                | StrOutputParser()
            )
        for _ in range(5):
            splits[-1].page_content = chain.invoke(splits[-1])
            last_chunk = splits[-1].page_content.split(' ')
            if max(Counter(last_chunk).values()) < 3: break

        with open(ewha_chunks_path, 'w') as f:
            for doc in splits:
                f.write(doc.json() + '\n')
    else:
        splits = []
        # If already exists, load the file from local
        with open(ewha_chunks_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                obj = Document(**data)
                splits.append(obj)
    print(f"[INFO] # of splits: {len(splits)}")
    #exit(-1)
    return splits 

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

def load_custom_dataset(dataset_name):
    """Load a custom dataset by name."""
    if dataset_name == "arc": # use arc dataset
        return load_arc()
    elif dataset_name == "other dataset": # new dataset(ex:cosmosqa)
        # add new load_dataset()
        return load_other_dataset(dataset_name)
    else:
        raise ValueError(f"[ERROR] Unsupported dataset: {dataset_name}") 
    
def load_other_dataset(dataset_name):
    """Loads *** dataset and make it as metadata"""
    ds = load_dataset(dataset_name)
    train_data = ds['train']
    train_docs = []
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

def get_chroma_vs(save_dir, embeddings, collection_name):
    vectorstore = Chroma(
            persist_directory=save_dir,
            embedding_function=embeddings,
            collection_name=collection_name,
            collection_metadata = {'hnsw:space': 'cosine'},
        )    
    return vectorstore
    
def get_MultiVecRetriever(vectorstore, store, id_key, top_k):
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": top_k},
    )
    return retriever

def get_chroma(splits, save_dir="./db/chroma", top_k=4, chunk_size=None, chunk_overlap=None, collection_name=""):
    embeddings = get_embedding() 
    if not os.path.exists(save_dir):
        os.mkdir(save_dir) 
        vectorstore = Chroma.from_documents(collection_name=collection_name, documents=splits, embedding=embeddings, persist_directory=save_dir) 
    else:
        # If the db already exists, load from local.
        vectorstore = get_chroma_vs(save_dir, embeddings, collection_name)
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    return retriever 

def get_summ_docs(splits, doc_ids, id_key):
    llm = get_llm(temperature=0)
    chain = (
        {"doc": lambda x: x.page_content}
        | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
        | llm
        | StrOutputParser()
    )
    summaries = chain.batch(splits, {"max_concurrency": 5})
    summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
    return summary_docs

def get_child(splits, doc_ids, child_text_splitter, id_key):
    data = dict()
    sub_docs = []
    for i, doc in enumerate(splits):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        data[doc.page_content] = [s.page_content for s in _sub_docs]
    sub_docs.extend(_sub_docs)
    return sub_docs, data

def get_pc_chroma_fix(splits, save_dir="./db/pc_chroma_fix", top_k=4, chunk_size=1000, chunk_overlap=100, debug=False):
    """Parent Document Retreiver using Chroma"""
    embeddings = get_embedding() 
    #docstore_path = os.path.join(save_dir, "docstore_pc.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    
    # The vectorstore to use to index the child chunks
    vectorstore = get_chroma_vs(save_dir, embeddings, "parent-child")

    # Layer to store parent document
    store = InMemoryByteStore()
    id_key = "doc_id"
    retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)

    
    # splitter to make chunk
    parent_text_splitter = RecursiveCharacterTextSplitter(
                chunk_overlap=100,
                chunk_size=800)

    child_text_splitter = RecursiveCharacterTextSplitter(
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size)

    splits = parent_text_splitter.split_documents(splits)
    doc_ids = [str(uuid.uuid4()) for _ in splits]

    sub_docs, data = get_child(splits, doc_ids, child_text_splitter, id_key)
    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, splits)))
    
    # Save as json file
    json_path = os.path.join(save_dir, f"./ewha_pc_{chunk_size}_{chunk_overlap}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii = False) 

    if debug:
        print("[DEBUG] Testing Parent-Child ...")
        retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "pc_chroma")
        retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "pc_chroma")
    return retriever

def get_pc_chroma(splits, save_dir="./db/pc_chroma", top_k=4, chunk_size=1000, chunk_overlap=100, debug=False):
    """Parent Document Retreiver using Chroma"""
    embeddings = get_embedding() 
    #docstore_path = os.path.join(save_dir, "docstore_pc.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    
    # The vectorstore to use to index the child chunks
    vectorstore = get_chroma_vs(save_dir, embeddings, "parent-child")

    # Layer to store parent document
    store = InMemoryByteStore()
    id_key = "doc_id"
    retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)

    doc_ids = [str(uuid.uuid4()) for _ in splits]

    # splitter to make child chunk
    child_text_splitter = RecursiveCharacterTextSplitter(
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size)
    
    sub_docs, data = get_child(splits, doc_ids, child_text_splitter, id_key)

    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, splits)))
    
    # Save as json file
    json_path = os.path.join(save_dir, f"./ewha_pc_{chunk_size}_{chunk_overlap}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii = False) 

    if debug:
        print("[DEBUG] Testing Parent-Child ...")
        retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "pc_chroma")
        retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "pc_chroma")
    return retriever

def get_summ_chroma(splits, save_dir="./db/summ_chroma", top_k=4, chunk_size=None, chunk_overlap=None, debug=True):
    """Parent Document Retreiver using Chroma with summarization"""
    embeddings = get_embedding() 
    docstore_path = os.path.join(save_dir, "docstore_summ.pkl")
    os.makedirs(save_dir, exist_ok=True) 

    # The vectorstore to use to index the child chunks
    vectorstore = get_chroma_vs(save_dir, embeddings, "summaries")
    # Layer to store parent document
    id_key = "doc_id"
    store = InMemoryByteStore()    

    if not os.path.exists(docstore_path):
        # The retriever (empty to start)
        retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)
        doc_ids = [str(uuid.uuid4()) for _ in splits]

        summary_docs = get_summ_docs(splits, doc_ids, id_key)

        retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, splits)))

        # Save the vectorstore and docstore to disk
        retriever.vectorstore.persist()
        with open(docstore_path, "wb") as file:
            pickle.dump(retriever.byte_store.store, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO] Successfully saved Vectorscore to local!")
    else:
        with open(docstore_path, "rb") as file:
            store_dict = pickle.load(file)
        store.mset(list(store_dict.items()))
        retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)
        print(f"[INFO] Load DB from {save_dir}...") 
    
    if debug:
        print("[DEBUG] Testing Parent-Child Summarization...")
        retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "summ_chroma")
        retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "summ_chroma")
    return retriever
        
def get_pc_faiss(splits, save_dir="./db/pc_faiss", top_k=4, chunk_size=1000, chunk_overlap=100, debug=True):
    embeddings = get_embedding() 

    docstore_path = os.path.join(save_dir, "docstore_pc.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    id_key = "doc_id"
    store = InMemoryByteStore()
    
    if not os.path.exists(docstore_path):
        doc_ids = [str(uuid.uuid4()) for _ in splits]

        # splitter to make child chunk
        child_text_splitter = RecursiveCharacterTextSplitter(
                    chunk_overlap=chunk_overlap,
                    chunk_size=chunk_size)
        
        sub_docs, data = get_child(splits, doc_ids, child_text_splitter, id_key)
        vectorstore = get_faiss_vs(sub_docs, embeddings)

        # The retriever (empty to start)
        retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)

        #retriever.vectorstore.add_documents(sub_docs)
        retriever.docstore.mset(list(zip(doc_ids, splits)))
        vectorstore.save_local(save_dir)
        with open(docstore_path, "wb") as file:
            pickle.dump(retriever.byte_store.store, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO] Successfully saved Vectorscore to local!")

        #retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, splits)))
        # Save the vectorstore and docstore to disk
        vectorstore.save_local(save_dir)
        with open(docstore_path, "wb") as file:
            pickle.dump(retriever.byte_store.store, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO] Successfully saved Vectorscore to local!")
        
    else:
        with open(docstore_path, "rb") as file:
            store_dict = pickle.load(file)
        store.mset(list(store_dict.items()))

        vectorstore = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True) 
        retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)
        print(f"[INFO] Load DB from {save_dir}...") 

    if debug:
        print("[DEBUG] Testing Parent-Child with FAISS...")
        retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "pc_faiss")
        retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "pc_faiss")
    return retriever
        

def get_summ_faiss(splits, save_dir="./db/summ_faiss", top_k=4, chunk_size=None, chunk_overlap=None, debug=False):
    embeddings = get_embedding() 
    docstore_path = os.path.join(save_dir, "docstore_summ.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    id_key = "doc_id"
    store = InMemoryByteStore()
    
    if not os.path.exists(docstore_path):
        doc_ids = [str(uuid.uuid4()) for _ in splits]
        summary_docs = get_summ_docs(splits, doc_ids, id_key)
        vectorstore = get_faiss_vs(summary_docs, embeddings)
        
        # The retriever (empty to start)
        retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)
    
        #retriever.vectorstore.add_documents(summary_docs)
        retriever.docstore.mset(list(zip(doc_ids, splits)))
        # Save the vectorstore and docstore to disk
        vectorstore.save_local(save_dir)
        with open(docstore_path, "wb") as file:
            pickle.dump(retriever.byte_store.store, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO] Successfully saved Vectorscore to local!")
    else:
        with open(docstore_path, "rb") as file:
            store_dict = pickle.load(file)
        store.mset(list(store_dict.items()))

        vectorstore = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True) 
        retriever = get_MultiVecRetriever(vectorstore, store, id_key, top_k)
        print(f"[INFO] Load DB from {save_dir}...") 
    
    if debug:
        print("[DEBUG] Testing Parent-Child with summarization using FAISS...")
        retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "summ_faiss")
        retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "summ_faiss")
    return retriever

def load_cross_encoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    """Load the cross-encoder model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return tokenizer, model

def re_rank_with_cross_encoder(query, docs, tokenizer, model, device="cpu"):
    """
    Re-rank documents using a cross-encoder model.
    Args:
        query: Search query string.
        docs: List of Document objects to rank.
        tokenizer: Cross-encoder tokenizer.
        model: Cross-encoder model.
        device: Device to run the model on ('cpu' or 'cuda').
    Returns:
        Ranked list of Document objects.
    """
    model.to(device)
    inputs = [
        tokenizer(
            query,
            doc.page_content,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        for doc in docs
    ]
    scores = []
    with torch.no_grad():
        for input_batch in inputs:
            input_batch = {k: v.to(device) for k, v in input_batch.items()}
            outputs = model(**input_batch)
            scores.append(outputs.logits.squeeze().item())

    # Sort documents by score (descending)
    ranked_docs = [doc for _, doc in sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)]
    return ranked_docs
 

def retriever_test(vectorstore, retriever, question, retriever_name):
    """Test the Parent-Child structure"""
    sub_docs = vectorstore.similarity_search(question)
    print(f"<Question> {question}")
    print(f"====== {retriever_name} child docs result ========")
    for i, doc in enumerate(sub_docs):
        processed_content = doc.page_content.replace('\n', ' ')
        print(f"[문서 {i}][{len(doc.page_content)}] {processed_content}")
    print()
    retrieved_docs = retriever.invoke(question)
    print(f"====== {retriever_name} parent docs result ========")
    for i, doc in enumerate(retrieved_docs):
        processed_content = doc.page_content.replace('\n', ' ')
        print(f"[문서 {i}][{len(doc.page_content)}] {processed_content}")
    print()
    print()

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
        chain = prompt_template | llm | StrOutputParser()
    return chain

def retrieve(db, query, tokenizer=None, model=None, device="cpu", use_reranking=False):
    """
    Perform a query on the retriever.
    If db is an EnsembleRetriever, use its invoke method.
    Otherwise, access the vectorstore directly.
    Optionally re-rank results using a cross-encoder.
    """
    if isinstance(db, EnsembleRetriever):
        docs = db.invoke(query)
    else:
        docs = db.vectorstore.similarity_search(query)
    
    # Apply Cross-encoder re-ranking if enabled
    if use_reranking and tokenizer and model:
        print("[INFO] Applying Cross-encoder re-ranking...")
        docs = re_rank_with_cross_encoder(query, docs, tokenizer, model, device)
    
    return docs

def grounded_check(context, answer):
    groundedness_check = UpstageGroundednessCheck()
    request_input = {
        "context": context,
        "answer": answer,
    }
    response = groundedness_check.invoke(request_input)
    #print(response)
    return response == "grounded" # grounded, notGrounded, or notSure

def get_pc_responses(db, chain, prompts, use_grounded):
    responses = []
    for prompt in prompts:
        context = retrieve(db, prompt)
        response = chain.invoke({"question": prompt, "context": context})
        if use_grounded:
            if not grounded_check(context, response.content) or "[ANSWER]:" not in response.content: # mmlu
                response = chain.invoke({"question": prompt, "context": context})
        responses.append(response) # response.content 
    return responses