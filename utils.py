import os
import re
import pandas as pd
import yaml


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def read_data(data_path, filename="test_samples.csv"):
    data = pd.read_csv(os.path.join(data_path, filename))
    prompts = data['prompts']
    answers = data['answers']
    # returns two lists: prompts and answers
    return prompts, answers 


def format_docs(docs):
    if 'question' in docs[0].metadata:  
            formatted_docs = "\n\n".join(doc.page_content for doc in docs)  
    else:
        formatted_docs = [doc.page_content for doc in docs] 
    return formatted_docs  

def format_arc_doc(data):
    question = data['question']
    choices = data['choices']['text']
    labels = data['choices']['label']
    
    formatted_choices = "\n".join([f"{label}: {choice}" for label, choice in zip(labels, choices)])
    formatted_doc = f"Question: {question}\nChoices:\n{formatted_choices}\n"
    
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


def eval(questions, answers, responses, debug=False):
    cnt = 0
    for question, answer, response in zip(questions, answers, responses):
        print("-"*10)
        print(f"{question}\n")
        generated_answer = extract_answer(response)
        if debug:
            print(f"[INFO][WHOLE RESPONSE]{response}")
            if generated_answer:
                print(f"\ngenerated answer: {generated_answer}, answer: {answer}")
            else:
                print("extraction fail")

        if generated_answer == None:
            continue
        if generated_answer in answer:
            cnt += 1

    accuracy = (cnt/len(answers))*100
    print(f"Accuracy: {accuracy}%")
    print("All Done") 
    return accuracy

