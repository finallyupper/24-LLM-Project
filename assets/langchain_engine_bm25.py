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
    os.chdir(ROOT)


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

def get_bm25(splits, save_dir="./db/bm25", top_k=4):
    # Where to save BM25
    bm25_path = os.path.join(save_dir, "bm25.pkl")

    # Load BM25 from local
    print("[INFO] Get retriever BM25 ...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(bm25_path):
        print("[INFO] Creating BM25 index...")
        # Make BM25
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

def get_ensemble_retriever(r1, r2, w1=0.7, w2=0.3):
    ensemble_retriever = EnsembleRetriever(
        retrievers=[r1, r2], weights=[w1, w2]
        )
    return ensemble_retriever 

def to_document(text: str, meta):
    return Document(id=meta, page_content=text, metadata={"p_id": meta})
                
def load_ewha(data_root):
    ewha_chunks_path = os.path.join(data_root, "ewha_chunk_doc.json")
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
        
        chain = (
                {"doc": lambda x: x.page_content}
                | ChatPromptTemplate.from_template("Please correct the spacing in the given text. It is a university regulation document. Just give only answer.\n\n{doc}")
                | ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=0)
                | StrOutputParser()
            )
        splits[-1].page_content = chain.invoke(splits[-1])

        with open(ewha_chunks_path, 'w', encoding='utf-8') as f:
            for doc in splits:
                f.write(doc.json() + '\n')
    else:
        splits = []
        with open(ewha_chunks_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                obj = Document(**data)
                splits.append(obj)
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

def get_pc_chroma(splits, save_dir="./db/pc_chroma", top_k=4, chunk_size=1000, chunk_overlap=100):
    embeddings = get_embedding() 
    docstore_path = os.path.join(save_dir, "docstore_pc.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    id_key = "doc_id"
    store = InMemoryByteStore()

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        persist_directory=save_dir,
        collection_name="parent-child",
        embedding_function=embeddings,
    )

    # The retriever (empty to start)
    retriever = MultiVectorRetriever(
        vectorstore=vectorstore,
        byte_store=store,
        id_key=id_key,
        search_kwargs={"k": top_k},
    )
    doc_ids = [str(uuid.uuid4()) for _ in splits]

    # child splitter
    child_text_splitter = RecursiveCharacterTextSplitter(
                chunk_overlap=chunk_overlap,
                chunk_size=chunk_size,
            )
    
    data = dict()
    sub_docs = []
    for i, doc in enumerate(splits):
        _id = doc_ids[i]
        _sub_docs = child_text_splitter.split_documents([doc])
        for _doc in _sub_docs:
            _doc.metadata[id_key] = _id
        data[doc.page_content] = [s.page_content for s in _sub_docs]
    sub_docs.extend(_sub_docs)
    retriever.vectorstore.add_documents(sub_docs)
    retriever.docstore.mset(list(zip(doc_ids, splits)))

    json_path = os.path.join(save_dir, f"./ewha_pc_{chunk_size}_{chunk_overlap}.json")
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4, ensure_ascii = False) 
    retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "pc_chroma")
    retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "pc_chroma")
    return retriever

def get_pc_faiss(splits, save_dir="./db/pc_faiss", top_k=4, chunk_size=1000, chunk_overlap=100):
    embeddings = get_embedding() 
    docstore_path = os.path.join(save_dir, "docstore_pc.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    id_key = "doc_id"
    store = InMemoryByteStore()
    
    if not os.path.exists(docstore_path):
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=0)
            | StrOutputParser()
        )
        doc_ids = [str(uuid.uuid4()) for _ in splits]
        summaries = chain.batch(splits, {"max_concurrency": 5})
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        vectorstore = FAISS.from_documents(summary_docs, embedding=embeddings)
        
        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": top_k},
        )
        doc_ids = [str(uuid.uuid4()) for _ in splits]

        # child splitter
        child_text_splitter = RecursiveCharacterTextSplitter(
                    chunk_overlap=chunk_overlap,
                    chunk_size=chunk_size,
                )
        
        data = dict()
        sub_docs = []
        for i, doc in enumerate(splits):
            _id = doc_ids[i]
            _sub_docs = child_text_splitter.split_documents([doc])
            for _doc in _sub_docs:
                _doc.metadata[id_key] = _id
            data[doc.page_content] = [s.page_content for s in _sub_docs]
        sub_docs.extend(_sub_docs)
        retriever.vectorstore.add_documents(sub_docs)
        retriever.docstore.mset(list(zip(doc_ids, splits)))
        vectorstore.save_local(save_dir)
        with open(docstore_path, "wb") as file:
            pickle.dump(retriever.byte_store.store, file, pickle.HIGHEST_PROTOCOL)
        print("[INFO] Successfully saved Vectorscore to local!")
    else:
        with open(docstore_path, "rb") as file:
            store_dict = pickle.load(file)
        store.mset(list(store_dict.items()))

        vectorstore = FAISS.load_local(save_dir, embeddings, allow_dangerous_deserialization=True) 
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": top_k},
        )
        print(f"[INFO] Load DB from {save_dir}...") 
    retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "pc_faiss")
    retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "pc_faiss")
    return retriever
        
def get_summ_chroma(splits, save_dir="./db/summ_chroma", top_k=4):
    embeddings = get_embedding() 
    docstore_path = os.path.join(save_dir, "docstore_summ.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    id_key = "doc_id"
    store = InMemoryByteStore()

    # The vectorstore to use to index the child chunks
    vectorstore = Chroma(
        persist_directory=save_dir,
        collection_name="summaries",
        embedding_function=embeddings,
    )

    if not os.path.exists(docstore_path):
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=0)
            | StrOutputParser()
        )
        summaries = chain.batch(splits, {"max_concurrency": 5})

        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": top_k},
        )
    
        doc_ids = [str(uuid.uuid4()) for _ in splits]
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]

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

        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": top_k},
        )
        print(f"[INFO] Load DB from {save_dir}...") 
    retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "summ_chroma")
    retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "summ_chroma")
    return retriever
        
def get_summ_faiss(splits, save_dir="./db/summ_faiss", top_k=4):
    embeddings = get_embedding() 
    docstore_path = os.path.join(save_dir, "docstore_summ.pkl")
    os.makedirs(save_dir, exist_ok=True) 
    id_key = "doc_id"
    store = InMemoryByteStore()
    
    if not os.path.exists(docstore_path):
        chain = (
            {"doc": lambda x: x.page_content}
            | ChatPromptTemplate.from_template("Summarize the following document:\n\n{doc}")
            | ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=0)
            | StrOutputParser()
        )
        doc_ids = [str(uuid.uuid4()) for _ in splits]
        summaries = chain.batch(splits, {"max_concurrency": 5})
        summary_docs = [
            Document(page_content=s, metadata={id_key: doc_ids[i]})
            for i, s in enumerate(summaries)
        ]
        vectorstore = FAISS.from_documents(summary_docs, embedding=embeddings)
        
        # The retriever (empty to start)
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": top_k},
        )
    
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
        retriever = MultiVectorRetriever(
            vectorstore=vectorstore,
            byte_store=store,
            id_key=id_key,
            search_kwargs={"k": top_k},
        )
        print(f"[INFO] Load DB from {save_dir}...") 
        
    retriever_test(vectorstore, retriever, "휴학은 최대 몇 년까지 할 수 있어?", "summ_faiss")
    retriever_test(vectorstore, retriever, "생활환경대학의 기존 이름은?", "summ_faiss")
    return retriever

def retriever_test(vectorstore, retriever, question, retriever_name):
    sub_docs = vectorstore.similarity_search(question)
    print(f"<Question> {question}")
    print(f"====== {retriever_name} child docs result ========")
    for i, doc in enumerate(sub_docs):
        doc_content = doc.page_content.replace('\n', ' ')
        print(f"[문서 {i}][{len(doc.page_content)}] {doc_content}")
    print()
    retrieved_docs = retriever.invoke(question)
    print(f"====== {retriever_name} docs result ========")
    for i, doc in enumerate(retrieved_docs):
        doc_content = doc.page_content.replace('\n', ' ')
        print(f"[문서 {i}][{len(doc.page_content)}] {doc_content}")
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
        chain = prompt_template | llm
    return chain

def retrieve(db, query):
    if isinstance(db, EnsembleRetriever):
        documents = db.get_relevant_documents(query)  # for EnsembleRetriever
    elif hasattr(db, "vectorstore"):
        documents = db.vectorstore.similarity_search(query)  # for individual FAISS/Chroma
    else:
        raise AttributeError(f"Unsupported retriever type: {type(db)}")
    
    # Convert Document objects to plain text
    context = "\n".join([doc.page_content for doc in documents])
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


def get_pc_responses(db, chain, prompts, use_grounded):
    responses = []
    for prompt in prompts:
        context = retrieve(db, prompt)  # Ensure context is a plain string
        response = chain.invoke({"question": prompt, "context": context})
        if use_grounded:
            if not grounded_check(context, response.content) or "[ANSWER]:" not in response.content:
                response = chain.invoke({"question": prompt, "context": context})
        responses.append(response.content)
    return responses


