import json
import os
import random
import re
import string
from prettytable import PrettyTable

import nltk
import numpy as np
import pandas as pd

from gensim.models import Word2Vec
import gensim.downloader as api

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_samples, silhouette_score

# Load pre-trained word2vec model
#wv = api.load('word2vec-google-news-300')

SEED = 42
random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)

### Operation settings
# 1: vectors
# 2: number of clusters
# 3: Mini-batch size
which_hyperparam = 0
which_task = 2

lamp2_vector_sizes = [140]
lamp2_cluster_numbers = [10]
lamp2_cluster_sizes = [32]

lamp4_vector_sizes = [220]
lamp4_cluster_numbers = [9]
lamp4_cluster_sizes = [32]

# Settings for hyperparams
if which_task == 2:
    vector_sizes = lamp2_vector_sizes # Settled on 140, maximized inertia
    cluster_numbers = lamp2_cluster_numbers
    cluster_sizes = lamp2_cluster_sizes # mb
else:
    vector_sizes = lamp4_vector_sizes
    cluster_numbers = lamp4_cluster_numbers
    cluster_sizes = lamp4_cluster_sizes

# Percentages of original corpus to maintain
percentage_to_keep = [.10, .20, .30, .40, .50, .60, .70, .80, .90]

# Tables from hyperparameter tuning
model_vector_tuning_headers = ['# of Model Vectors', 'Inertia', 'Silhouette coefficient']
num_clusters_headers = ['# of Clusters', 'Inertia', 'Silhouette coefficient']
mini_batch_headers = ['Mini-batch Size', 'Inertia', 'Silhouette coefficient']

# Find more stable metrics of the number of clusters
if which_hyperparam == 1:
    average_inertias = {k : [] for k in vector_sizes}
    average_silhouette = {k : [] for k in vector_sizes}
elif which_hyperparam == 2:
    average_inertias = {k : [] for k in cluster_numbers}
    average_silhouette = {k : [] for k in cluster_numbers}
elif which_hyperparam == 3:
    average_inertias = {k : [] for k in cluster_sizes}
    average_silhouette = {k : [] for k in cluster_sizes}

def generate_models(path, task):
    ## load the pre-processed documents 
    with open(path, 'r') as file:
        training_data = json.load(file)
        
    # Train model using list of all of the documents
    tokenized_all_docs = []
    for id in training_data:
        user_profile = training_data[id][1]
        documents = [f"{item['title']} | {item['text']}" for item in user_profile]
        tokenized_documents = [doc.split(" ") for doc in documents]
        tokenized_all_docs.extend(tokenized_documents)
        
    print(f"All documents in training data tokenized: {len(tokenized_all_docs)}")
    for size_of_vector in vector_sizes:
        print(f"begining training of Word2Vec model of size {size_of_vector}")
        model = Word2Vec(sentences=tokenized_all_docs, vector_size=size_of_vector, workers=2, seed=SEED)
        model.save(f"models/lamp{task}/{size_of_vector}.bin")


# list_of_docs: list of tokenized documents
def vectorize(list_of_docs, model):
    """Generate vectors for list of documents using a Word Embedding

    Args:
        list_of_docs: List of documents
        model: Gensim's Word Embedding

    Returns:
        List of document vectors
    """
    features = []

    for tokens in list_of_docs:
        zero_vector = np.zeros(model.vector_size)
        vectors = []
        for token in tokens:
            if token in model.wv:
                try:
                    vectors.append(model.wv[token])
                except KeyError:
                    continue
        if vectors:
            vectors = np.asarray(vectors)
            avg_vec = vectors.mean(axis=0)
            features.append(avg_vec)
        else:
            features.append(zero_vector)
    return features


# Returns a dictionary containing the vectorized documents for each user profile for the associated ID
def word_2_vec(path, model_path, minimum_profile_size=0):

    # load model from bin
    model = Word2Vec.load(model_path)
    
    # Load the data that you want vectorized
    with open(path, 'r') as file:
        data = json.load(file)
    
    vectorized_documents_id = dict()
    
    # Reverse lookup table {document string : document ID}
    reverse_document_id_lookup = dict()
    
    total_user_profiles = len(data)
    current = 1
    
    docs = dict()
    tokenized_docs = dict()
    
    for id in data:
        # print(f"{current}/{total_user_profiles}")
        current += 1
        user_profile = data[id][1]
        
        # Skip user profiles with not enough documents to cluster        
        if len(user_profile) < minimum_profile_size:
            continue

        # Tokenize the list of documents
        documents = []
        for item in user_profile:
            document_string = f"{item['title']} {item['text']}"
            documents.append(document_string)
            # Depending on which task, we need to encapsulate different data in the reverse lookup
            if which_task == 2:
                reverse_document_id_lookup[document_string] = {'title' : item['title'],
                                                             'text'  : item['text'],
                                                             'category' : item['category'],
                                                            'id'    : item['id']}
            else:
                reverse_document_id_lookup[document_string] = {'title' : item['title'],
                                                                'text'  : item['text'],
                                                                'id'    : item['id']}
            
        tokenized_documents = [doc.split(" ") for doc in documents]
        
        vectorized_docs = vectorize(tokenized_documents, model=model)
        
        vectorized_documents_id[id] = vectorized_docs
        docs[id] = documents
        tokenized_docs[id] = tokenized_documents
        
    return vectorized_documents_id, tokenized_docs, docs, reverse_document_id_lookup

def mbkmeans_clusters(
	X, 
    k, 
    mb, 
    print_silhouette_values,
    # For hyperparameter testing
    hyperparameter_tested,
    model_vector_size
):
    """Generate clusters and print Silhouette metrics using MBKmeans

    Args:
        X: Matrix of features.
        k: Number of clusters.
        mb: Size of mini-batches.
        print_silhouette_values: Print silhouette values per cluster.

    Returns:
        Trained clustering model and labels based on X.
    """
    km = MiniBatchKMeans(n_clusters=k, batch_size=mb).fit(X)
    # print(f"For n_clusters = {k}")
    # print(f"Silhouette coefficient: {silhouette_score(X, km.labels_):0.2f}")
    # print(f"Inertia:{km.inertia_}")

    #### Add metrics to tested hyperparameters ####
    if hyperparameter_tested == 1: # Model Vectors
        average_inertias[model_vector_size].append(km.inertia_)
        average_silhouette[model_vector_size].append(float(f"{silhouette_score(X, km.labels_):0.2f}"))
    elif hyperparameter_tested == 2: # number of clusters
        average_inertias[k].append(km.inertia_)
        average_silhouette[k].append(float(f"{silhouette_score(X, km.labels_):0.2f}"))
    elif hyperparameter_tested == 3: # mini-batch size
        average_inertias[mb].append(km.inertia_)
        average_silhouette[mb].append(float(f"{silhouette_score(X, km.labels_):0.2f}"))
    else:
        pass

    if print_silhouette_values:
        sample_silhouette_values = silhouette_samples(X, km.labels_)
        print(f"Silhouette values:")
        silhouette_values = []
        for i in range(k):
            cluster_silhouette_values = sample_silhouette_values[km.labels_ == i]
            silhouette_values.append(
                (
                    i,
                    cluster_silhouette_values.shape[0],
                    cluster_silhouette_values.mean(),
                    cluster_silhouette_values.min(),
                    cluster_silhouette_values.max(),
                )
            )
        silhouette_values = sorted(
            silhouette_values, key=lambda tup: tup[2], reverse=True
        )
        for s in silhouette_values:
            print(
                f"    Cluster {s[0]}: Size:{s[1]} | Avg:{s[2]:.2f} | Min:{s[3]:.2f} | Max: {s[4]:.2f}"
            )
    return km, km.labels_


def main():
    
    # Test performance of the different models with different amounts of vectors
    for size_of_vector in vector_sizes: # settled on 80 for vector size
    
        print(f"Vector Size: {size_of_vector}")
    
        # Find the document vectors for the preprocessed dataset
        user_document_vectors, tokenized_docs, docs, reverse_lookup = word_2_vec(f"processed_data/lamp{which_task}/validate/questions.json", f"models/lamp{which_task}/{size_of_vector}.bin", minimum_profile_size=100)
        
        which_id = 1
        # Form the clusters based on the document vectors
        for id in user_document_vectors.keys():
            print(f"{which_id}/{len(user_document_vectors)}")
            which_id += 1
            
            for cluster_size in cluster_sizes:
                
                for cluster_num in cluster_numbers:
                    #print(f"# of clusters: {cluster_num}")
                    clustering, cluster_labels = mbkmeans_clusters(
                    X=user_document_vectors[id],
                    k=cluster_num,
                    mb=cluster_size,
                    print_silhouette_values=False,
                    hyperparameter_tested=which_hyperparam, # Hyperparameter testing
                    model_vector_size=size_of_vector
                    )
                    df_clusters = pd.DataFrame({
                        "text": docs[id],
                        "tokens": [" ".join(text) for text in tokenized_docs[id]],
                        "cluster": cluster_labels
                    })
                    
                    # begin formulating the reduced corpus with the clustering and cluster labels:
                    
                    
                    
        print("--------------------------------------------------------------------------------------------------------------------------")

    # Save the results of hyperparameter tuning
    ## Vector size
    if which_hyperparam == 1:
        vector_size_table = PrettyTable()
        vector_size_table.field_names = model_vector_tuning_headers
        for key in sorted(average_inertias.keys()):
            row = [key, 
                sum(average_inertias[key]) / len(average_inertias[key]), 
                sum(average_silhouette[key]) / len(average_silhouette[key])]
            vector_size_table.add_row(row)
        vector_size_table_string = vector_size_table.get_string()
        with open(f"tuning/lamp{which_task}/Model_vector_tuning.txt", 'w') as file:
            file.write(vector_size_table_string)
    
    # ## Number of clusters
    elif which_hyperparam == 2:
        num_clusters_table = PrettyTable()
        num_clusters_table.field_names = num_clusters_headers
        # Average metrics
        for key in sorted(average_inertias.keys()):
            row = [key, 
                sum(average_inertias[key]) / len(average_inertias[key]), 
                sum(average_silhouette[key]) / len(average_silhouette[key])]
            num_clusters_table.add_row(row) 
        num_clusters_table_string = num_clusters_table.get_string()
        with open(f"tuning/lamp{which_task}/num_clusters_tuning.txt", 'w') as file:
            file.write(num_clusters_table_string)
    
    # ## Mini batch size
    elif which_hyperparam == 3:
        mini_batch_table = PrettyTable()
        mini_batch_table.field_names = mini_batch_headers
        for key in sorted(average_inertias.keys()):
            row = [key, 
                sum(average_inertias[key]) / len(average_inertias[key]), 
                sum(average_silhouette[key]) / len(average_silhouette[key])]
            mini_batch_table.add_row(row)
        mini_batch_table_string = mini_batch_table.get_string()
        with open(f"tuning/lamp{which_task}/mini_batch_tuning.txt", 'w') as file:
            file.write(mini_batch_table_string)

if __name__ == '__main__':
    main()