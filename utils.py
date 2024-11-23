import os
import re
import pandas as pd
import yaml

MULTI_RETRIEVAL_ROUTER_TEMPLATE = """
    Given the input, choose the most appropriate model prompt based on the provided prompt descriptions.

    "Prompt Name": "Prompt Description"

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the retrieve to use or "DEFAULT"
        "next_inputs": string \ an original version of the original input
    }}}}
    ```

    REMEMBER: "destination" should be chosen based on the descriptions of the available prompts, or "DEFAULT" if no appropriate prompt is found.
    REMEMBER: "next_inputs" MUST be the original input.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""

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
    wrong_questions = []
    for question, answer, response in zip(questions, answers, responses):
        print("-"*10)
        print(f"{question}\n")
        generated_answer = extract_answer(response)
        if debug:
            if len(response) > 601:
                response = response[:600]
            print(f"[Total Response]{response}")
            if generated_answer:
                print(f"\ngenerated answer: {generated_answer}, answer: {answer}")
            else:
                print("extraction fail")

        if generated_answer == None:
            continue
        if generated_answer in answer:
            cnt += 1
        else:
            wrong_question = re.findall(r"QUESTION(\d+)", question)
            wrong_questions.append(wrong_question)
    accuracy = (cnt/len(answers))*100
    print(f"Accuracy: {accuracy}%")
    print("All Done") 
    print(f"Wrong Answers in these questions:", end=" ")
    for q in wrong_questions:
        print(f"{q[0]},", end=" ") 

    return accuracy

def document_to_dict(doc):
    return {
        "metadata": doc.metadata,
        "page_content": doc.page_content,
    }