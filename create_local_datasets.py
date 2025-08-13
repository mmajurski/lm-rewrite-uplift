import json
import pandas as pd
from datasets import load_dataset
import tqdm



def squad():
    # Load the SQuAD dataset from Huggingface
    print("Loading SQuAD dataset from Huggingface...")
    squad_dataset = load_dataset("squad")
    # squad_dataset["train"] = squad_dataset["train"].select(range(100))
    # squad_dataset["validation"] = squad_dataset["validation"].select(range(100))

    # Convert the dataset to a list of dictionaries
    print("Converting dataset to JSON format...")
    train_data = []
    for i in range(len(squad_dataset["train"])):
        train_data.append({
            "question": squad_dataset["train"][i]["question"],
            "answer": squad_dataset["train"][i]["answers"]["text"][0],
            "context": squad_dataset["train"][i]["context"]
        })
    for i in range(len(squad_dataset["validation"])):
        train_data.append({
            "question": squad_dataset["validation"][i]["question"],
            "answer": squad_dataset["validation"][i]["answers"]["text"][0],
            "context": squad_dataset["validation"][i]["context"]
        })
    print(f"Combined train and validation sets: {len(train_data)} examples")

    # Save the datasets to JSON files
    print(f"Saving training set ({len(train_data)} examples)...")
    with open("squadv2.json", "w") as f:
        json.dump(train_data, f, indent=2)


def triva_qa():
    print("Loading dataset from Huggingface...")
    lcl_dataset = load_dataset("mandarjoshi/trivia_qa", 'rc')["train"]

    # Convert the dataset to a list of dictionaries
    print("converting dataset to list of dictionaries...")
    train_data = []
    for i in range(len(lcl_dataset)):
        context = ""
        search_results = lcl_dataset[i]["search_results"]
        for c in search_results['search_context']:
            context += c + "\n\n"

        train_data.append({
            "question": lcl_dataset[i]["question"],
            "answer": lcl_dataset[i]["answer"]['value'],
            "context": context
        })

    # Save the datasets to JSON files
    print(f"Saving training set ({len(train_data)} examples)...")
    with open("triva_qa.json", "w") as f:
        json.dump(train_data, f, indent=2)




# https://huggingface.co/datasets/mrqa-workshop/mrqa
def mrqa():
    print("Loading dataset from Huggingface...")
    lcl_dataset = load_dataset("mrqa-workshop/mrqa")

    # Convert the dataset to a list of dictionaries
    print("finding what subsets exist...")
    subsets = list(set(list(lcl_dataset["train"]['subset'])))
    subsets.extend(list(set(list(lcl_dataset["validation"]['subset']))))
    subsets = list(set(subsets))
    print(f"found {len(subsets)} subsets")
    dat = dict()
    for subset in subsets:
        dat[subset] = []
    print("converting dataset to list of dictionaries...")
    orig_data = lcl_dataset["train"]
    
    for i in tqdm.tqdm(range(len(orig_data))):
            
        cur_data = orig_data[i]
        subset = cur_data['subset']
        dat[subset].append({
            "question": cur_data["question"],
            "answer": cur_data['answers'][0],
            "context": cur_data["context"]
        })
    
    for sub in dat:
        print(f"Saving data set ({len(dat[sub])} examples)...")
        with open(f"mrqa_{sub}.json", "w") as f:
            json.dump(dat[sub], f, indent=2)

 





def pubmed_qa():
    print("Loading dataset from Huggingface...")
    dataset = load_dataset("qiaojin/PubMedQA", "pqa_labeled")

    # Convert the dataset to a list of dictionaries
    print("Converting dataset to JSON format...")
    train_data = []
    for i in range(len(dataset["train"])):
        context = ""
        for c in dataset["train"][i]["context"]['contexts']:
            context += c + "\n\n"
        train_data.append({
            "question": dataset["train"][i]["question"],
            "answer": dataset["train"][i]["final_decision"],
            "context": context,
            "long_answer": dataset["train"][i]["long_answer"]
        })

    # Save the datasets to JSON files
    print(f"Saving training set ({len(train_data)} examples)...")
    with open("pubmed_qa.json", "w") as f:
        json.dump(train_data, f, indent=2)



def natural_questions():
    print("Loading dataset from Huggingface...")
    dataset = load_dataset("google-research-datasets/natural_questions")

    # Convert the dataset to a list of dictionaries
    print("Converting dataset to JSON format...")
    train_data = []
    for i in tqdm.tqdm(range(len(dataset["train"]))):
        val = dataset["train"][i]
        q = val['question']['text']
        a = val['annotations']['short_answers'][0]['text']

        if isinstance(a, list) and len(a) > 0:
            a = a[0]
        else:
            a = None
        if a is None:
            continue
        d = val['document']['html']
        if isinstance(d, list):
            d = d[0]

        train_data.append({
            "question": q,
            "answer": a,
            "context": d,
        })

    train_data_top = train_data[:100]
    with open("natural_questions_top.json", "w") as f:
        json.dump(train_data_top, f, indent=2)

    # Save the datasets to JSON files
    print(f"Saving training set ({len(train_data)} examples)...")
    with open("natural_questions.json", "w") as f:
        json.dump(train_data, f, indent=2)


def flashrag_2wikimultihopqa():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(a) > 1:
                a = " or ".join(val['golden_answers'])
            context = ""
            for c in val["metadata"]['context']['content']:
                for c2 in c:
                    context += c2 + "\n\n"
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_2wikimultihopqa.json", "w") as f:
        json.dump(train_data, f, indent=2)


def flashrag_boolq():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "boolq")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(val['golden_answers']) > 1:
                a = " or ".join(val['golden_answers'])
            context = val["metadata"]['passage']
            
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_boolq.json", "w") as f:
        json.dump(train_data, f, indent=2)


def flashrag_fermi():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "fermi")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev", "test"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(val['golden_answers']) > 1:
                a = " or ".join(val['golden_answers'])
            context = val["metadata"]['context']
            
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_fermi.json", "w") as f:
        json.dump(train_data, f, indent=2)
    

def flashrag_hotpotqa():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(val['golden_answers']) > 1:
                a = " or ".join(val['golden_answers'])
            context = ""
            for c in val["metadata"]['context']['sentences']:
                for c2 in c:
                    context += c2 + "\n\n"
            
            
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_hotpotqa.json", "w") as f:
        json.dump(train_data, f, indent=2)
    


def flashrag_msmarcoqa():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "msmarco-qa")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(a) > 1:
                a = " or ".join(val['golden_answers'])
            context = ""
            for c in val["metadata"]['passages']['passage_text']:
                context += c + "\n\n"
            
            
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_msmarcoqa.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    
def flashrag_musique():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "musique")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(val['golden_answers']) > 1:
                a = " or ".join(val['golden_answers'])
            context = ""
            for c in val["metadata"]['question_decomposition']:
                context += c['support_paragraph']['paragraph_text'] + "\n\n"
            
            
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_musique.json", "w") as f:
        json.dump(train_data, f, indent=2)


def flashrag_narrativeqa():
    dataset = load_dataset("RUC-NLPIR/FlashRAG_datasets", "narrativeqa")
    print("Converting dataset to JSON format...")
    train_data = []
    for split in ["train", "dev"]:
        for i in tqdm.tqdm(range(len(dataset[split]))):
            val = dataset[split][i]
            q = val['question']
            a = val['golden_answers'][0]
            if len(val['golden_answers']) > 1:
                a = " or ".join(val['golden_answers'])
            context = val["metadata"]['text']
            
            
            train_data.append({
                "question": q,
                "answer": a,
                "context": context,
            })

    print(f"Saving training set ({len(train_data)} examples)...")
    with open("flashrag_narrativeqa.json", "w") as f:
        json.dump(train_data, f, indent=2)
    
    
    
    



if __name__ == "__main__":
    # squad()
    # triva_qa()
    # mrqa()
    # pubmed_qa()
    
    # natural_questions()
    # flashrag_2wikimultihopqa()
    # flashrag_boolq()
    # flashrag_fermi()
    # flashrag_hotpotqa()
    # flashrag_msmarcoqa()
    # flashrag_musique()
    flashrag_narrativeqa()