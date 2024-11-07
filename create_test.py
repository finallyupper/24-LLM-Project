"""
Last Updated: 24/11/08
"""
import os
import parmap
import pandas as pd
import multiprocessing
from dotenv import load_dotenv
from datasets import load_dataset
from datasets import concatenate_datasets

def extract_row(row):
    with idx.get_lock():
            idx.value += 1
    prompt = f'QUESTION{idx.value}) {row['question']}\n'
    for i in range(len(row['options'])):
        opt = chr(ord('A') + i) 
        prompt += f"({opt}) {row['options'][i]}\n"
    answer = f"({row['answer']})"
    mnmn.append({"prompts": prompt.rstrip('\n'), "answers": answer})

def main():
    load_dotenv(verbose=True)
    ROOT = os.getenv('ROOT')
    given = pd.read_csv(os.path.join(ROOT, "test_samples.csv")) # # of samples is 10
    ds = load_dataset("TIGER-Lab/MMLU-Pro", cache_dir=ROOT)
    ds = concatenate_datasets([ds['validation'], ds['test']])

    global mnmn, idx
    manager = multiprocessing.Manager()
    n_proc = multiprocessing.cpu_count()
    mnmn = manager.list()
    idx = multiprocessing.Value('i', len(given))
    
    with multiprocessing.Pool(n_proc, initargs = (idx, )) as pool:
        parmap.map(
            extract_row, 
            ds, 
            pm_pbar=True, pm_processes=n_proc)
    assert len(mnmn)==len(ds)

    df = pd.concat([given, pd.DataFrame(list(mnmn))])
    df.to_csv(os.path.join(ROOT, "test_samples_v2.csv"), index=False, sep=',')
    print("DataFrame Shape:", df.shape)
    print("All Done")

if __name__=="__main__":
    main()