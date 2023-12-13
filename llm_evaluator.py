import numpy as np
import pandas as pd
import nltk
import evaluate
import torch
import transformers as tf
import json
import re
import time
import random
import copy
from llm_tuning import ForT5Dataset


def json_load(path):
    """
    Quick helper to dump JSON files to python objects.
    Args:
        path: string file path of the json file.

    Returns: Contained JSON object.
    """
    with open(path, 'r') as file:
        out = json.load(file)
        file.close()
    return out


class LLM_Evaluator:
    def __init__(self, lamp2_path, lamp4_path, global_debug=False):
        self.lamp2_model, self.lamp2_tokenizer, self.lamp4_model, self.lamp4_tokenizer = self.load_llms(lamp2_path,
                                                                                                        lamp4_path)
        nltk.download("punkt", quiet=True)
        self.rouge_ev = evaluate.load("rouge")
        self.debug = global_debug
        self.labels_path = ""

    @staticmethod
    def load_llms(lamp2_path, lamp4_path):
        """
        Populates the model/tokenizer variables
        Args:
            lamp2_path: string file path to the tuned lamp2 model directory
            lamp4_path: string file path to the tuned lamp4 model directory

        Returns:
            Model and tokenizer objects for both models

        """
        lamp2_model = tf.T5ForConditionalGeneration.from_pretrained(lamp2_path)
        lamp2_tokenizer = tf.T5Tokenizer.from_pretrained(lamp2_path)

        lamp4_model = tf.T5ForConditionalGeneration.from_pretrained(lamp4_path)
        lamp4_tokenizer = tf.T5Tokenizer.from_pretrained(lamp4_path)

        return [lamp2_model, lamp2_tokenizer, lamp4_model, lamp4_tokenizer]

    def rouge(self, preds, labels):
        """
        Calculates and returns the rouge score for a list of predicted strings and true labels.
        Wrapper to handle the way this rouge package expects newlines.
        Args:
            preds: array[string], predicted labels.
            labels: array[string], true labels.

        Returns:
            rouge: float mean rouge for the dataset.
        """
        split_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in preds]
        split_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in labels]
        rouge_metric = self.rouge_ev.compute(predictions=split_preds, references=split_labels, use_stemmer=True)
        return rouge_metric

    def lamp2_preprocess(self, questions):
        """
        Slightly modified from the tuning version to include all given profile history and remove labels
        Args:
            questions: Questions dataset
        Returns: Tokenized dataset
        """
        history_prefix = "This user's recent history is: "
        history_suffix = " Using this information for context only, "
        prefix = ["Which category does this article relate to among the following categories? Just answer with the "
                  "category name without further explanation. categories: [women, religion, politics, style & beauty, "
                  "entertainment, culture & arts, sports, science & technology, travel, business, crime, education, "
                  "healthy living, parents, food & drink"]
        inputs = []
        question_vec = [questions[key][0] for key in questions.keys()]
        profiles = [questions[key][1] for key in questions.keys()]
        for i in range(len(question_vec)):
            profile = profiles[i]
            title_cats = [item['title'] + ": " + item['category'] + ". " for item in profile]
            random.shuffle(title_cats)

            model_input = history_prefix + "".join(title_cats[:20]) + history_suffix + str(prefix) + question_vec[i]
            inputs.append(model_input)
            if len(model_input) > 512 and self.debug:
                print(len(model_input))

        inputs = self.lamp2_tokenizer(inputs, max_length=512, padding=True, truncation=True, return_tensors="pt")

        return inputs

    def lamp4_preprocess(self, questions):
        """
        Slightly modified from the tuning version to include all given profile history and remove labels
        Args:
            questions: Questions json dataset

        Returns: Tokenized dataset
        """

        history_prefix = "This user's recent stemmed titles include: "
        history_suffix = ". Using this information for context, "

        inputs = []

        fulltext = json_load(self.labels_path + "questions.json")
        i = 0
        for question in fulltext:
            profile = question['profile']
            red_ids = [doc['id'] for doc in questions[question['id']][1]]
            full_doc_ids = {}
            for item in profile:
                full_doc_ids[item['id']] = item['title']
            idx = []
            for red_id in red_ids:
                title = full_doc_ids[red_id]
                idx.append(title)
                #print(title)
            random.shuffle(idx)
            staged_input = history_prefix + ". ".join(idx[:10])
            model_input = staged_input[:min(100,len(staged_input))] + history_suffix + question['input']
            inputs.append(model_input)
            if len(model_input) > 512:
                print(len(model_input))
            i += 1

        # for i in range(len(questions)):
        #     question = question_vec[i]
        #     profile = profiles[i]
        #     ids = [item['id'] for item in profile]
        #
        #     title_cats = [item['title'] for item in profile]
        #     random.shuffle(title_cats)
        #
        #     model_input = history_prefix + "".join(title_cats[:20]) + history_suffix + str(prefix) + question['input']
        #     inputs.append(model_input)

        inputs = self.lamp4_tokenizer(inputs, max_length=512, truncation=False, padding=True, return_tensors="pt")

        return inputs

    def lamp2_eval(self, inputs, outputs) -> dict:
        """
        Test and evaluate the summarization model
        Args:
            inputs: Tokenized article/profile string array
            outputs: list[string] true category array
        Returns:
            {'rouge': mean rouge score (string similarity),
            'acc': total accuracy}
        """

        pred_outputs = self.lamp2_model.generate(**inputs, max_new_tokens=10)
        answer = map(self.lamp2_tokenizer.decode, pred_outputs)
        # l_answer = list(answer)
        # if len(l_answer) == 0:
        #     print(answer, l_answer)

        pred_outputs = []
        total = 0
        correct = 0
        for item in answer:
            answer = re.sub("<pad> *([a-zA-Z]*[ &]*[a-zA-Z]*)\\.*.*", "\\1", item)
            pred_outputs.append(answer)

            if len(answer) != 0 and self.debug:
                print(answer)

            if answer.lower() == outputs[total]:
                correct += 1

            total += 1

        mean_rouge = self.rouge(pred_outputs, outputs)

        acc = correct / total

        return {'rouge': mean_rouge, 'acc': acc}

    def lamp4_eval(self, inputs, outputs) -> dict:
        """
        Test and evaluate the summarization model
        Args:
            inputs: list[string] article/profile string array
            outputs: list[string] test summary array
        Returns:
            {'rouge': mean rouge score (string similarity),
            'acc': total accuracy}
        """
        pred_outputs = self.lamp4_model.generate(**inputs)
        answer = map(self.lamp4_tokenizer.decode, pred_outputs)
        answers = []
        total, correct = 0,0
        for item in answer:
            answer = re.sub("<pad> *([a-zA-Z& ])\\.*.*", "\\1", item)
            answers.append(answer)
            if answer.lower() == outputs[total]:
                correct += 1
            total += 1


        mean_rouge = self.rouge(answers, outputs)
        acc = correct/total

        return {'rouge': mean_rouge, 'acc': acc}

    def evaluate_reduced_corpus(self, corpus_path, labels_path, task="class", file_stem="question_0."):
        """
        Import model, import corpus, run folded evals on each step of corpus, display results
        Args:
            corpus_path: string, file path of the root directory of the reduced corpus
            labels_path: string, file path of the true label file
            task: string, which model to use. "class" for LAMP2, "summ" for LAMP4.
            file_stem: string, unchanging file prefix. (i.e. question_0.1.json to question_0.9.json)

        Returns:
            Optional numpy array of rouge scores and accuracy metrics, probably. If it doesn't just print them.
        """
        if task == "class":
            model = self.lamp2_model
            tokenizer = self.lamp2_tokenizer
            preprocess_fn = self.lamp2_preprocess
            eval_fn = self.lamp2_eval
        elif task == "summ":
            model = self.lamp4_model
            tokenizer = self.lamp4_tokenizer
            preprocess_fn = self.lamp4_preprocess
            eval_fn = self.lamp4_eval
        else:
            raise ValueError("%s is not a valid model task, must be one of 'class' or 'summ'" % task)
        self.labels_path = labels_path
        # load array of the reduced corpus questions
        datasets = []
        print("Loading data at file path %s..." % corpus_path)
        for i in range(9):
            dataset = json_load(corpus_path + file_stem + str(i + 1) + ".json")
            print(len(dataset.keys()))
            datasets.append(dataset)

        print("Data loading complete.")
        # Preprocessing the labels
        raw_labels = json_load(labels_path + "outputs.json")['golds']

        labels = [output['output'] for output in raw_labels]
        print(len(labels))
        print("Preprocessing...")
        processed_datasets = list(map(preprocess_fn, datasets))
        print("Preprocessing complete.\nEvaluating...")
        metrics = []

        for i in range(9):
            start = time.time()
            out = eval_fn(processed_datasets[i], labels)
            diff = time.time() - start
            print(out)
            print("Fold %d of 9 processed in %.2f seconds." % (i, diff))
            metrics.append(out)

        return
