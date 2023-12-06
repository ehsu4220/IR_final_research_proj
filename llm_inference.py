import numpy as np
import pandas as pd
import nltk
import evaluate
import torch
import transformers as tf
import json


lamp2_tuned = "./tuned_llms/lamp2"
tokenizer = tf.T5Tokenizer.from_pretrained(lamp2_tuned)
lamp2_model = tf.T5ForConditionalGeneration.from_pretrained(lamp2_tuned)

base_prefix = ("Which category does this article relate to among the following categories? Just answer with the "
               "category name without further explanation. categories: [women, religion, politics, style & beauty, "
               "entertainment, culture & arts, sports, science & technology, travel, business, crime, education, "
               "healthy living, parents, food & drink]")

history_prefix = "This user's recent history is: "
history_suffix = " Using this information for context only, "


test_article = ("article: The spa industry remains relentlessly apathetic and unapologetically uneducated in caring "
                "for the skin of 80 percent of the world's population.")

test_history = ("Five Spa Tips Every Twenty-Something Should Know. Category: style & beauty."
                "Why Aren't Spas on TripAdvisor?. Category: travel."
                "Is Your Spa Ready for a Walk-Through?. Category: style & beauty."
                "What Game of Thrones Can Teach Us About Spa Retail Training. Category: business.")

test_input = history_prefix + test_history + history_suffix + base_prefix + test_article

inputs = tokenizer(test_input, return_tensors="pt")
outputs = lamp2_model.generate(**inputs)
answer = tokenizer.decode(outputs[0])
print(answer)
