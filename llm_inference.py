import numpy as np
import pandas as pd
import nltk
import evaluate
import torch
import transformers as tf
import json
from llm_evaluator import LLM_Evaluator
# cats = ["women, religion, politics, style & beauty, "
#               "entertainment, culture & arts, sports, science & technology, travel, business, crime, education, "
#               "healthy living, parents, food & drink]


evaluator = LLM_Evaluator("tuned_llms/lamp2","tuned_llms/lamp4",False)
stats_l2 = evaluator.evaluate_reduced_corpus("reduced_corpus/lamp2/validate/", "data/lamp2/validate/","class")
stats_l4 = evaluator.evaluate_reduced_corpus("reduced_corpus/lamp4/validate/", "data/lamp4/validate/", "summ")


