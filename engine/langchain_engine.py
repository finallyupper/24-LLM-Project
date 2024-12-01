from __future__ import annotations
from dotenv import load_dotenv
import os
import re
import threading
import itertools
import time
from tqdm import tqdm 
import random
from typing import Any, Dict, List, Optional

from datasets import load_dataset
from engine.utils import *
from prompts import MULTI_RETRIEVAL_ROUTER_TEMPLATE

from langchain import hub
from langchain_upstage import ChatUpstage, UpstageEmbeddings, UpstageLayoutAnalysisLoader
from langchain_text_splitters import Language,RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain.chains import ConversationChain
from langchain.chains.base import Chain
from langchain.chains.conversation.prompt import DEFAULT_TEMPLATE
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_retrieval_qa import MultiRetrievalQAChain

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

# Returns upstage embedding
def get_embedding():
    """Loads upstage embedding"""
    
    print("[INFO] Loading embeddings...")
    return UpstageEmbeddings(api_key=UPSTAGE_API_KEY, model = 'solar-embedding-1-large')

def get_llm(temperature=0):
    """Loads LLM model from Upstage"""
    llm = ChatUpstage(api_key = UPSTAGE_API_KEY, temperature=temperature)
    return llm 

class newMultiRetQAChain(MultiRetrievalQAChain):
    """
    A multi-route chain that uses an LLM router chain to choose amongst retrieval qa chains.
    
    The change in the existing MultiRetrievalQAChain class:
    When passing the prompt to the destination chain, 
    the test sample's question is now transmitted without any modification and remains unchanged.
    """
    @classmethod
    def from_retrievers(
        cls,
        llm: BaseLanguageModel,
        retriever_infos: List[Dict[str, Any]],
        default_retriever: Optional[BaseRetriever] = None,
        default_prompt: Optional[PromptTemplate] = None,
        default_chain: Optional[Chain] = None,
        *,
        default_chain_llm: Optional[BaseLanguageModel] = None,
        **kwargs: Any,
    ) -> newMultiRetQAChain:
        if default_prompt and not default_retriever:
            raise ValueError(
                "`default_retriever` must be specified if `default_prompt` is "
                "provided. Received only `default_prompt`."
            )
        destinations = [f"{r['name']}: {r['description']}" for r in retriever_infos]
        destinations_str = "\n".join(destinations)

        router_template = MULTI_RETRIEVAL_ROUTER_TEMPLATE.format(
            destinations=destinations_str,
        )

        router_prompt = PromptTemplate(
            template=router_template,
            input_variables=["input"],
            output_parser=RouterOutputParser(next_inputs_inner_key="query"),
        )
        router_chain = LLMRouterChain.from_llm(llm, router_prompt)
        destination_chains = {}
        for r_info in retriever_infos:
            prompt = r_info.get("prompt")
            retriever = r_info["retriever"]
            
            chain = RetrievalQA.from_llm(llm, prompt=prompt, return_source_documents=True, retriever=retriever)
            name = r_info["name"]
            destination_chains[name] = chain
        if default_chain:
            _default_chain = default_chain
        elif default_retriever:
            _default_chain = RetrievalQA.from_llm(
                llm, prompt=default_prompt, return_source_documents=True, retriever=default_retriever
            )
        else:
            prompt_template = DEFAULT_TEMPLATE.replace("input", "query")
            prompt = PromptTemplate(
                template=prompt_template, input_variables=["history", "query"]
            )
            if default_chain_llm is None:
                raise NotImplementedError(
                    "conversation_llm must be provided if default_chain is not "
                    "specified. This API has been changed to avoid instantiating "
                    "default LLMs on behalf of users."
                    "You can provide a conversation LLM like so:\n"
                    "from langchain_openai import ChatOpenAI\n"
                    "llm = ChatOpenAI()"
                )
            _default_chain = ConversationChain(
                llm=default_chain_llm,
                prompt=prompt,
                input_key="query",
                output_key="result",
            )
        return cls(
            router_chain=router_chain,
            destination_chains=destination_chains,
            default_chain=_default_chain,
            **kwargs,
        )

def get_router(llm, retrievers, prompt_template=None, verbose=True):
    """A multi-route chain that uses an LLM router chain to choose amongst retrieval
    qa chains."""

    if prompt_template is not None:
         prompt_template1 = PromptTemplate.from_template(prompt_template[0]) # EWHA 
         prompt_template2 = PromptTemplate.from_template(prompt_template[1]) # Common for MMLU
         prompt_template3 = PromptTemplate.from_template(prompt_template[2]) # BASE
    else:
        prompt_template1 = hub.pull("rlm/rag-prompt") #QA prompt  https://smith.langchain.com/hub/rlm/rag-prompt?organizationId=5b2073af-2123-4ed3-b218-fa406e467d84 
        prompt_template3 = prompt_template2 = prompt_template1
        
    retriever_infos = [
        {
            "name": "ewha_retriever",
            "description": "for 이화여자대학교 학칙(Rules of Ewha Womans University); 학교(대학)의 학과, 교과 과정, 입학, 졸업, 상벌, 총칙, 부설기관, 학사 운영, 학점, 성적, 학생 활동 및 행정 절차 등에 관한 규칙. A university is an institution of higher (or tertiary) education (undergraduate and postgraduate programs) and research which awards academic degrees in several academic disciplines(majors).",
            "retriever": retrievers[0],
            "prompt": prompt_template1
        },
        {
            "name": "law_retriever",
            "description": "An expert for law; a set of rules that are created and are enforceable by social or governmental institutions to regulate behavior,[1] with its precise definition a matter of longstanding debate. law is a system of rules established by governing authorities to regulate behavior, maintain order, and resolve disputes. Not related with university rules",
            "retriever": retrievers[1][0],
            "prompt": prompt_template2
        },
        {
            "name": "psychology_retriever",
            "description": "An expert for psychology; the scientific study of mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. psychology is the scientific study of the mind and behavior, exploring how individuals think, feel, and act. Not related with university rules",
            "retriever": retrievers[1][1],
            "prompt": prompt_template2
        },
        {
            "name": "philosophy_retriever",
            "description": "An expert for philosophy; a systematic study of general and fundamental questions concerning topics like existence, reason, knowledge, value, mind, and language. It is a rational and critical inquiry that reflects on its own methods and assumptions. Philosophy is the study of fundamental questions regarding existence, knowledge, ethics, and reason. Not related with university rules",
            "retriever": retrievers[1][2],
            "prompt": prompt_template2
        },
        {
            "name": "business_retriever",
            "description": "An expert for business; the practice of making one's living or making money by producing or buying and selling products (such as goods and services). It is also 'any activity or enterprise entered into for profit.' business involves the creation, management, and operation of organizations that provide goods or services for profit. Not related with university rules",
            "retriever": retrievers[1][3],
            "prompt": prompt_template2
        },
        {
            "name": "history_retriever",
            "description": "An expert for history; the systematic study and documentation of the human past. history is the study of past events and societies, examining how they have shaped the present and future. Human history is the record of humankind from prehistory to the present. Not related with university rules",
            "retriever": retrievers[1][4], 
            "prompt": prompt_template2
        },
    ]

    default_retriever = retrievers[-1]
    
    multi_retrieval_qa_chain = newMultiRetQAChain.from_retrievers(
        llm=llm,
        retriever_infos=retriever_infos,
        default_retriever=default_retriever,
        default_prompt=prompt_template3,
        verbose=True
    )
    return multi_retrieval_qa_chain 

def get_option(question, response, debug=False, eval=False):
    answer = get_answers(response)
    if extract_answer(answer, eval) is not None: return answer
    if 'Answer:\n' in answer:
        answer = answer.replace("Answer:\n", f"[ANSWER]: ")
        print("[INFO] EUREKA:", "[ANSWER]: "+ answer.split('[ANSWER]:')[-1].strip())
    else:
        pattern = r'\(([A-Z])\)\s(.*?)[\u2028\n]'
        options = re.findall(pattern, question)
        print("[INFO] OPTIONS:", options)
        candidate = answer.split('[ANSWER]:')[-1].strip()
        for (question_alphabet, question_text) in options:
            if debug:
                print(f"Option {question_alphabet}:")
                print(" ", question_text.strip())
                print(" ", answer.split('[ANSWER]:')[-1].strip())
            if candidate.startswith(question_text.strip()) \
                or ''.join(candidate.split(' ')).startswith(''.join(question_text.strip().split(' '))):
                answer = answer.replace("[ANSWER]:", f"[ANSWER]: ({question_alphabet}) {question_text}")
                print("[INFO] EUREKA:", "[ANSWER]: "+ answer.split('[ANSWER]:')[-1].strip())
                return answer
    answer = answer.replace("\u2028", "\n")
    return answer

# Get LLM's answer regardless of format
def get_answers(response):
    if isinstance(response, str): answer = response
    else:
        try: answer = response['result']
        except: answer = response.content
    return answer

# Collect final responses with Safeguard
def get_responses(chain, safeguard, prompts, debug=False):
    responses = []
    for prompt in tqdm(prompts, desc="Processing questions"):
        try:
            response = chain.invoke(prompt) 
            if debug: print("[INFO] ROUTE: ", response['result'])
            if extract_answer(response['result']) is None: # RFV(Response Format Validation, Extracting the Answer)
                response['result'] = get_option(prompt, response)
                
            # Ewha Safeguard
            if len(response['source_documents']) > 1 and extract_answer(get_answers(response), eval=True) is None: # RFV
                response = safeguard[0].invoke(prompt)
                response['result'] = get_option(prompt, response)
                if debug: print("[INFO] EWHA SAFEGUARD: ", response)
            # MMLU Safeguard
            if len(response['source_documents']) < 2 or \
                (isinstance(response, dict) and extract_answer(get_answers(response), eval=True) is None): # RFV
                response = safeguard[1].invoke(prompt)
                response = get_option(prompt, response)
                if debug: print("[INFO] MMLU SAFEGUARD: ", response)
        # When router chooses non-exist destination chain
        ## ex) ValueError: Received invalid destination chain name 'education_retriever'
        except ValueError: 
            response = safeguard[0].invoke(prompt) # Ewha Retriever
            response = get_option(prompt, response)
            if debug: print("[INFO] SAFEGUARD: ", response)
        responses.append(get_answers(response))
    return responses

def get_faiss_vs(splits, embeddings):
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings) 
    return vectorstore 
       
def get_faiss(splits, save_dir="./db/ewha/ewha_faiss_fix", chunk_size=None, chunk_overlap=None, top_k=4, thres=0.8): 
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

    retriever = vectorstore.as_retriever(
                    search_type="similarity_score_threshold", 
                    search_kwargs={"score_threshold": thres, 
                    "k": top_k}) # Modified(Su)
    return retriever 

def load_customed_datasets(type):
    train_docs = []
    if type == "law": # 24.4k
        print("[INFO] Load ymoslem/Law-StackExchange dataset...")
        ds_law = load_dataset("ymoslem/Law-StackExchange")
        train_data_law = ds_law['train'] 
        sample_size = 2000 
        train_data_list = list(train_data_law)
        sampled_data_law = random.sample(train_data_list, sample_size) 

        for entry in tqdm(sampled_data_law):
            if len(entry['answers']) == 0:
                continue 
            doc_content = format_law_docs(entry) 
            doc = Document(page_content=doc_content,
                            metadata={"question": entry['question_title'], 
                                      "details": entry['question_body'], 
                                      "answers": entry["answers"]})
            train_docs.append(doc) 
        return train_docs  
    
    elif type == "psychology": # 197k
        print("[INFO] Load BoltMonkey/psychology-question-answer dataset...")
        ds_psy = load_dataset("BoltMonkey/psychology-question-answer")
        train_data_psy = ds_psy['train'] 
        sample_size = 2000 # Only use 2.0k because of memory issue 

        train_data_list = list(train_data_psy)
        sampled_data_psy = random.sample(train_data_list, sample_size)

        for entry in tqdm(sampled_data_psy):
            doc_content = format_psy_docs(entry) 
            doc = Document(page_content=doc_content,
                           metadata={"question": entry['question'],
                                     "answer": entry['answer']}) 
            train_docs.append(doc) 
        return train_docs 
    
    elif type == "business": # 1.02k
        print("[INFO] Load Rohit-D/synthetic-confidential-information-injected-business-excerpts dataset ...")
        ds_bis = load_dataset("Rohit-D/synthetic-confidential-information-injected-business-excerpts")
        train_data_bis = ds_bis['train']
        for entry in tqdm(train_data_bis):
            doc_content = format_bis_docs(entry)
            doc = Document(page_content=doc_content,
                           metadata={"excerpt": entry['Excerpt'],
                                     "reason": entry["Reason"]})
            train_docs.append(doc) 
        return train_docs 
    
    elif type == "philosophy": # 134k
        print("[INFO] Load sayhan/strix-philosophy-qa dataset ...")
        ds_phi = load_dataset("sayhan/strix-philosophy-qa") 
        train_data_phi = ds_phi['train'] 
        sample_size = 2000 # Only use 1.8k because of memory issue --> 2.0k 4004tokens error(> 4000tokens limitation)
        train_data_list = list(train_data_phi)
        sampled_data_phi = random.sample(train_data_list, sample_size)

        for entry in tqdm(sampled_data_phi):
            doc_content = format_phi_docs(entry) 
            doc = Document(page_content = doc_content,
                           metadata= {"category": entry['category'],
                                      "question": entry['question'],
                                      "answer": entry['answer']})
            train_docs.append(doc) 
        return train_docs 
    
    elif type == "history":
        print("[INFO] Load nielsprovos/world-history-1500-qa dataset...") 
        ds_hist = load_dataset("nielsprovos/world-history-1500-qa")
        train_data_hist = list(ds_hist['train'])[0]['qa_pairs'] # 376
        for entry in tqdm(train_data_hist):
            doc_content = format_hist_docs(entry)
            doc = Document(page_content = doc_content,
                            metadata={"question": entry['question'],
                                      "answer": entry['answer']})

            train_docs.append(doc) 
        return train_docs
    
    assert len(train_docs) !=0, "Input correct type!"

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
