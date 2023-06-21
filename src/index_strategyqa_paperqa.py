from load_datasets import load_strategyqa
from tqdm import tqdm
import re
from paperqa import Docs
from constant_variables import PATH_STRATEGYQA_DOCS
import pickle 

def index_strategyqa_docs():
    docs = Docs()
    train_paragraphs = load_strategyqa(PATH_STRATEGYQA_DOCS)

    all_chunks = [{'source' : identifier, 
                'title' : data['title'], 
                'content' : data['content']} 
                for identifier, data in train_paragraphs.items()]
    
    return_msg = ''
    for chunk in tqdm(all_chunks):
        page_id = re.sub(
                r"[^\w\d]", "_", f"{chunk['source']}"
            )
        file_full_path = '../documents/BingSearch_DBStrategyQA/' + page_id + '.txt'
        with open(file_full_path, "w", encoding='utf-8') as f:
            f.write(chunk['content'])

        if file_full_path in docs.docs:
            print("This Page already in the Docs")
        else:
            print(f"Indexing Page : {page_id}")

            try:
                docs.add(file_full_path, citation=page_id, key=page_id)
                return_msg += f'Found and Indexed Page {page_id}\n'

            except Exception as e:
                print(f"Failed to index Page: {page_id}. Error Message: {e}")
                return_msg += f'Found but could not Index Page {page_id}\n'

    with open('VectorStore/embeddings.pkl', "wb") as f:
        pickle.dump(docs, f)
    return_msg


if __name__ == '__main__':
    return_msg = index_strategyqa_docs()
    print(return_msg)