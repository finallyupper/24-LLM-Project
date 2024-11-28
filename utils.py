import os
import re
import random
import numpy as np
import pandas as pd
import yaml
from bs4 import BeautifulSoup


def clean_html(html_text):
    soup = BeautifulSoup(html_text, 'html.parser')
    return soup.get_text()

def load_yaml(file_path: str) -> dict:
    """Loads configurations from yaml file"""
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def read_data(data_path, filename="test_samples.csv"):
    """Reads data and returns two lists"""
    data = pd.read_csv(os.path.join(data_path, filename))
    prompts = data['prompts']
    answers = data['answers']
    # returns two lists: prompts and answers
    print(f"[INFO] We got {len(prompts)} test samples")
    return prompts, answers 


def format_docs(docs):
    """formatting function used in chain"""
    if 'question' in docs[0].metadata:  
            formatted_docs = "\n\n".join(doc.page_content for doc in docs)  
    else:
       formatted_docs = [doc.page_content for doc in docs] 

    return formatted_docs  

def format_arc_doc(data):
    """Defines the format of arc dataset that is used for loading dataset"""
    question = data['question']
    choices = data['choices']['text']
    labels = data['choices']['label']
    
    formatted_choices = "\n".join([f"{label}: {choice}" for label, choice in zip(labels, choices)])
    formatted_doc = f"Question: {question}\nChoices:\n{formatted_choices}\n"
    return formatted_doc

def format_law_docs(data): # one row
    question = data['question_title']
    _contexts = data['question_body']
    _answers = data['answers'] 

    formatted_questions = []
    contexts = clean_html(_contexts)
    def get_best_answer(answers):
        # Sort answers by score in descending order
        if len(answers) == 0:
            print("HERE!")
        best_answer = max(answers, key=lambda x: x['score'])
        return best_answer['body'] 

    best_answer = clean_html(get_best_answer(_answers))
    formatted_doc = f"### Question: {question}\nDetails: {contexts}\nBest Answer: \n{best_answer}\n"
    return formatted_doc

def format_psy_docs(data):
    question = data['question'] 
    answer = data['answer'] 
    formatted_doc = f"### Question: {question}\nAnswer:\n{answer}\n"
    return formatted_doc

def format_bis_docs(data):
    excerpt = data['Excerpt']
    reason = data['Reason']
    formatted_doc = f"### Excerpt: {excerpt}\nReason:\n{reason}\n"
    return formatted_doc 

def format_phi_docs(data):
    category = data['category']
    question = data['question']
    answer = data['answer'] 
    formatted_doc = f"### Category: {category}\nQuestion: {question}\nAnswer:{answer}\n"
    return formatted_doc

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

def random_select(question):
    pattern = r"\((A|B|C|D|E)\)"  # Regular expression to capture the answer letter and text
    print(re.findall(pattern, question)[0])
    match = np.unique(list(re.findall(pattern, question)[0]))
    if match:
        num = random.randint(0, len(match)-1)
        return match[num] # Extract the letter inside parentheses (e.g., A)
    else:
        return random.choice(["A", "B"])

def eval(questions, answers, responses, debug=False):
    cnt_total = cnt_ewha = cnt_mmlu = 0
    total_questions = len(answers)
    wrong_questions_total, wrong_questions_ewha, wrong_questions_mmlu = [], [], []

    ewha_indices = list(range(35))
    mmlu_indices = list(range(35, total_questions))

    for i, (question, answer, response) in enumerate(zip(questions, answers, responses)):
        print("-"*10)
        print(f"Question {i + 1}: {question}\n")
        generated_answer = extract_answer(response)
        if debug:
            print(f"[Total Response]{response}")
            if generated_answer:
                print(f"\ngenerated answer: {generated_answer}, answer: {answer}")
            else:
                print("extraction fail")
                generated_answer = random_select(question)
                print(f"{generated_answer} selected")
        try:
            is_correct = generated_answer in answer
        except: is_correct = False

        # Overall query
        if is_correct:
            cnt_total += 1
        else:
            wrong_questions_total.append(i + 1)

        # Ewha query
        if i in ewha_indices:
            if is_correct:
                cnt_ewha += 1
            else:
                wrong_questions_ewha.append(i + 1)
        
        # MMLU-pro query
        elif i in mmlu_indices:
            if is_correct:
                cnt_mmlu += 1
            else:
                wrong_questions_mmlu.append(i + 1)
    
    accuracy_total = (cnt_total / total_questions) * 100
    accuracy_ewha = (cnt_ewha / (cnt_ewha + len(wrong_questions_ewha))) * 100 if cnt_ewha + len(wrong_questions_ewha) > 0 else 0
    accuracy_mmlu = (cnt_mmlu / (cnt_mmlu + len(wrong_questions_mmlu))) * 100 if cnt_mmlu + len(wrong_questions_mmlu) > 0 else 0

    print(f"Overall Accuracy: {accuracy_total:.2f}%")
    print(f"Ewha Accuracy: {accuracy_ewha:.2f}%")
    print(f"MMLU-pro Accuracy: {accuracy_mmlu:.2f}%")

    print(f"Wrong Answers (Overall): {wrong_questions_total}")
    print(f"Wrong Answers (Ewha): {wrong_questions_ewha}")
    print(f"Wrong Answers (MMLU-pro): {wrong_questions_mmlu}")

    return accuracy_total, accuracy_ewha, accuracy_mmlu
    

def document_to_dict(doc):
    return {
        "metadata": doc.metadata,
        "page_content": doc.page_content,
    }



