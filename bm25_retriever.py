from rank_bm25 import BM25Okapi

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# Helper function that handled normalization, stemming, and stopwords
def preprocess_text(text):
    tokens = word_tokenize(text.lower())  # Convert to lowercase and tokenize
    filtered_tokens = [stemmer.stem(word) for word in tokens if word.isalnum() and word not in stop_words]
    return ' '.join(filtered_tokens)

def preprocess_user_data(document_list):
    for doc in document_list:
        doc['text'] = preprocess_text(doc['text'])
        doc['title'] = preprocess_text(doc['title'])
        # print(f"{doc['text']} : {doc['title']}")

# Process the questions and extract the user profiles, associated inputs, and IDs
def process_questions(path, target_path):
    with open(path, 'r') as file:
        data = json.load(file)

    # Process the json
    processed_data = {item['id']: [item['input'], item['profile']] for item in data}
    
    # Normalize input and user profile
    for id in processed_data.keys():
        processed_data[id][0] = preprocess_text(processed_data[id][0].split("article: ", 1)[1])
        #print(f"Input prompt: {processed_data[id][0]}")
        preprocess_user_data(processed_data[id][1])

    # Save into file
    with open(target_path, 'w') as json_file:
        json.dump(processed_data, json_file)
        
    
    return processed_data


# Process the outputs and formulate the IDs and associated outputs
def process_outputs(path):
    with open(path, 'r') as file:
        data = json.load(file)
        
    # Process the json
    outputs = data['golds']
    processed_data = {item['id']: item['output'] for item in outputs}
    
    # Normalize output values
    
    return processed_data


# BM25 retriever model
def retrieve_relevant_k_docs(input, user_profile, k=20, minimum_profile_size=100):

    if k > minimum_profile_size or len(user_profile) < minimum_profile_size:
        return -1

    # Tokenize the list of documents
    documents = [f"{item['title']} {item['text']}" for item in user_profile]
    tokenized_documents = [doc.split(" ") for doc in documents]
    
    # Generate reverse lookup dictionary for converting document back to ID
    reverse_document_lookup = dict()
    for item in user_profile:
        reverse_document_lookup[f"{item['title']} {item['text']}"] = (item['id'], item['category'])
    
    # Tokenize the input query
    tokenized_input = input.split(" ")
    
    # Retriever
    bm25 = BM25Okapi(tokenized_documents)
    
    # retrieve documents
    retrieved_documents = bm25.get_top_n(tokenized_input, documents, k)
    
    # return the IDs associated with top k
    return [reverse_document_lookup[key][0] for key in retrieved_documents if key in reverse_document_lookup]
    
    


def main():
    
    # Processes the questions: stopwords, normalization, stemming
    ## Results are saved into json files in processed_data/ in order to avoid the need to constantly process this.
    #process_questions("data/lamp2/train/questions.json", "processed_data/lamp2/train/questions.json")
    #process_questions("data/lamp4/train/questions.json", "processed_data/lamp4/train/questions.json")
    #process_questions("data/lamp2/validate/questions.json", "processed_data/lamp2/validate/questions.json")
    #process_questions("data/lamp4/validate/questions.json", "processed_data/lamp4/validate/questions.json")
    process_questions("data/lamp2/test/questions.json", "processed_data/lamp2/test/questions.json")
    process_questions("data/lamp4/test/questions.json", "processed_data/lamp4/test/questions.json")

    # # Outputs
    # lamp2_outputs = process_outputs("data/lamp2/train/outputs.json")
    # lamp4_outputs = process_outputs("data/lamp4/train/outputs.json")
    
    # load the preprocessed user profiles
    # with open("processed_data/lamp2/train/questions.json", 'r') as lamp2_train:
    #     data = json.load(lamp2_train)
    
    # for id in data:
        # retrieved_documents_ids = retrieve_relevant_k_docs(data[id][0], data[id][1])
        
        # Retrieve the text associated with the profile and save somewhere for later fine-tuning on the LLM
    
    # Form (input, output) pairs.
    print("debug")

if __name__ == '__main__':
    main()