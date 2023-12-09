import numpy as np
import pandas as pd
import nltk
import evaluate
import torch
import transformers as tf
import json
from llm_evaluator import LLM_Evaluator

evaluator = LLM_Evaluator("tuned_llms/lamp2","tuned_llms/lamp4")
evaluator.evaluate_reduced_corpus("reduced_corpus/lamp4/train/", "data/lamp4/train/outputs.json")

