from rank_bm25 import BM25Okapi

import metrics

import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import matplotlib.pyplot as plt

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

# code parameters
current_task = 2
min_profile_size = 25

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
        print(len(user_profile))
        return -1, -1, -1

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
    
    # Get all relevancy scores and generate lookup table for relevancy scores
    document_relevancy = bm25.get_scores(tokenized_input)
    document_relevance_index = dict()
    for i in range(len(documents)):
        document_relevance_index[documents[i]] = document_relevancy[i]
    
    # return the IDs associated with top k
    return [reverse_document_lookup[key][0] for key in retrieved_documents if key in reverse_document_lookup], retrieved_documents, document_relevance_index


def main():
    
    # Processes the questions: stopwords, normalization, stemming
    ## Results are saved into json files in processed_data/ in order to avoid the need to constantly process this.
    #process_questions("data/lamp2/train/questions.json", "processed_data/lamp2/train/questions.json")
    #process_questions("data/lamp4/train/questions.json", "processed_data/lamp4/train/questions.json")
    #process_questions("data/lamp2/validate/questions.json", "processed_data/lamp2/validate/questions.json")
    #process_questions("data/lamp4/validate/questions.json", "processed_data/lamp4/validate/questions.json")
    # process_questions("data/lamp2/test/questions.json", "processed_data/lamp2/test/questions.json")
    # process_questions("data/lamp4/test/questions.json", "processed_data/lamp4/test/questions.json")

    # # Outputs
    # lamp2_outputs = process_outputs("data/lamp2/train/outputs.json")
    # lamp4_outputs = process_outputs("data/lamp4/train/outputs.json")
    
    # load the preprocessed user profiles
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    dcg_percentage = dict()
    
    for percentage in percentages:
        with open(f"processed_data/lamp{current_task}/test/questions.json", 'r') as whole_corpus:
            data = json.load(whole_corpus)
        
        with open(f"reduced_corpus/lamp{current_task}/test/question_{percentage}.json", 'r') as reduced_corpus:
            reduced_data = json.load(reduced_corpus)
        
        k_values = [5, 10, 15, 20]
        
        # Retrieve the text associated with the profile and generate metric results
        dcg_k = dict()
        for k in k_values:
            dcg_k[k] = []
            for id in reduced_data:
                _, whole_retrieved_doc_ids, whole_relevant_scores = retrieve_relevant_k_docs(data[id][0], data[id][1], k=k, minimum_profile_size=min_profile_size)
                _, reduced_retrieved_doc_ids, reduced_relevant_scores = retrieve_relevant_k_docs(reduced_data[id][0], reduced_data[id][1], k=k, minimum_profile_size=min_profile_size)
                
                if whole_retrieved_doc_ids == -1 or reduced_retrieved_doc_ids == -1:
                    print("retrieval failed")
                    continue
                
                # Perform the comparison of the DCG
                result = metrics.reduced_corpus_dcg_ratio(reduced_relevant_scores, whole_relevant_scores, reduced_retrieved_doc_ids, whole_retrieved_doc_ids)
                if result == -1:
                    print("metric failed")
                    continue
                else:
                    dcg_k[k].append(result)
                    
        # Average out each list of DCG ratios
        for key in dcg_k:
            dcg_k[key] = sum(dcg_k[key]) / len(dcg_k[key])
        print("debug")
        
        dcg_percentage[percentage] = dcg_k
    
    # Generate graph for the different plots
    # plt.figure()
    # for percentage in percentages:
    #     x_values = list(dcg_percentage[percentage].keys())
    #     y_values = list(dcg_percentage[percentage].values())
    #     plt.plot(x_values, y_values, marker='o', linestyle='-', label=percentage)
    
    # plt.xlabel('Top K')
    # plt.ylabel('DCG ratio')
    # plt.title(f'DCG performance of Varying Sized User Profiles: Minimum Size = {min_profile_size}')
    # plt.legend()
    
    # plt.ylim(0, 1)
    
    # plt.grid(True)
    # plt.show()
    
    k_values = set(k_val for percentages in dcg_percentage.values() for k_val in percentages.keys())
    
    plt.figure()
    
    for k in k_values:
        x_values = list(dcg_percentage.keys())
        y_values = [percentages[k] for percentages in dcg_percentage.values()]
        
        plt.plot(x_values, y_values, marker='o', label=f'k={k}')
        
    plt.xlabel('Percentage of Original Corpus')
    plt.ylabel('DCG Ratio')
    plt.title(f'DCG performance of Varying Sized User Profiles: Minimum Size = {min_profile_size}')
    plt.legend()
    
    plt.ylim(0, 1)
    plt.xlim(0, 1)
    plt.grid(True)
    plt.show()
        

if __name__ == '__main__':
    main()