import math

# Retrieve the document results and evaluate how diverse the retrieval results are
def reduced_corpus_dcg_ratio(reduced_corpus_scores, entire_corpus_scores, reduced_top_k, entire_top_k):
    # Ensure that the length of the top k documents are the same length
    if len(reduced_top_k) != len(entire_top_k) or len(reduced_top_k) == 0:
        print("Unequal top k")
        return -1
    
    reduced_dcg = 0
    entire_dcg = 0
    for i in range(len(reduced_top_k)):
        added_reduce = reduced_corpus_scores[reduced_top_k[i]]
        added_entire = entire_corpus_scores[entire_top_k[i]]
        reduced_dcg += added_reduce / math.log(i + 2, 2)
        entire_dcg += added_entire / math.log(i + 2, 2)
    
    if reduced_dcg == 0:
        print("something wrong, debug")
    # return the ratio of DCGs
    if entire_dcg == 0: # result of the NaN values
        return -1
    else:
        return reduced_dcg / entire_dcg