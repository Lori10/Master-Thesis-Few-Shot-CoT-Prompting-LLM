from langchain.vectorstores import FAISS
from load_datasets import load_strategyqa
from tqdm import tqdm
from langchain.embeddings.openai import OpenAIEmbeddings
from constant_variables import INDEX_PATH, PATH_STRATEGYQA_DOCS

def index_docs(batch_limit=100):      
    train_paragraphs = load_strategyqa(PATH_STRATEGYQA_DOCS)

    all_chunks = [{'source' : identifier, 
                   'title' : data['title'], 
                   'content' : data['content']} 
                   for identifier, data in train_paragraphs.items()]
        
    first_chunk = all_chunks[0]
    first_metadata = {key : first_chunk[key] for key in first_chunk if key in ['source', 'title']}
    db = FAISS.from_texts(texts=[first_chunk['content']], embedding=OpenAIEmbeddings(), metadatas=[first_metadata])

    batch_text_data = []
    batch_meta_data = []
    
    for i, example in enumerate(tqdm(all_chunks[1:])):
        text_data = example['content']
        meta_data = {key : example[key] for key in example if key in ['source', 'title']}

        batch_text_data.append(text_data)
        batch_meta_data.append(meta_data)
        
        try:
            if len(batch_text_data) >= batch_limit:                
                db.add_texts(batch_text_data, batch_meta_data)
                batch_text_data = []
                batch_meta_data = []

            elif i == len(all_chunks) - 2:
                if batch_text_data != [] and batch_meta_data != []:
                    db.add_texts(batch_text_data, batch_meta_data)
        except Exception as e:
            print(f'Could not index the batch: {batch_meta_data}')
            print(f'Error Message: {e}')
    
    db.save_local(INDEX_PATH)
    return 'Indexing done!'

if __name__ == '__main__':
    return_msg = index_docs()
    print(return_msg)

