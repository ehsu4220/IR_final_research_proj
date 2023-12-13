import numpy as np
import pandas as pd
import nltk
import evaluate
import torch
import transformers as tf
import json

MODEL_NAME = "google/flan-t5-base"
tokenizer = tf.T5Tokenizer.from_pretrained(MODEL_NAME)
llm2_model = tf.T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
llm4_model = tf.T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
llm2_collator = tf.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=llm2_model)
llm4_collator = tf.DataCollatorForSeq2Seq(tokenizer=tokenizer, model=llm4_model)

N_PROFILE = 5


class ForT5Dataset:
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.inputs["input_ids"][index]).squeeze()
        target_ids = torch.tensor(self.targets["input_ids"][index]).squeeze()

        return {"input_ids": input_ids, "labels": target_ids}


def llm2_preprocess(questions, labels):
    history_prefix = "This user's recent history is: "
    history_suffix = " Using this information for context only, "
    inputs = []
    labels = [output['output'] for output in labels]
    for question in questions:
        profile = question['profile']
        title_cats = [item['title'] + ". Category: " + item['category'] + ". " for item in profile]
        model_input = history_prefix + "".join(title_cats[:N_PROFILE]) + history_suffix + question['input']
        inputs.append(model_input)

    inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(text_target=labels, max_length=512, truncation=True)

    return ForT5Dataset(inputs, labels)

def llm4_preprocess(questions, labels):
    history_prefix = "This user's recent titles were: "
    history_suffix = ". Using this information for context only, "
    inputs = []
    labels = [output['output'] for output in labels]
    for question in questions:
        profile = question['profile']
        titles = ["title: " + item['title'] for item in profile]
        model_input = history_prefix + ". ".join(titles[:N_PROFILE]) + history_suffix + question['input']
        inputs.append(model_input)
        #print(model_input)

    inputs = tokenizer(inputs, max_length=512, truncation=True)
    labels = tokenizer(text_target=labels, max_length=512, truncation=True)

    return ForT5Dataset(inputs, labels)


# Scoring
nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")


def evaluate_metrics(predictions):
    preds, labels = predictions
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]
    rouge_metric = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return rouge_metric

# Helper fn for the data loads
def json_file_load(path):
    with open(path, 'r') as file:
        out = json.load(file)
        file.close()
    return out


l2out = json_file_load("data/lamp2/train/outputs.json")
l2in = json_file_load("data/lamp2/train/questions.json")
outputs = l2out['golds']

l2Vout = json_file_load("data/lamp2/validate/outputs.json")
l2Vin = json_file_load("data/lamp2/validate/questions.json")
test_outputs = l2Vout['golds']

l4out = json_file_load("data/lamp4/train/outputs.json")
l4in = json_file_load("data/lamp4/train/questions.json")
l4outputs = l4out['golds']


l4Vout = json_file_load("data/lamp4/validate/outputs.json")
l4Vin = json_file_load("data/lamp4/validate/questions.json")
l4Voutputs = l4Vout['golds']

preprocessed_data_train = llm2_preprocess(l2in, outputs)
preprocessed_data_test = llm2_preprocess(l2Vin, test_outputs)

llm4_preprocessed_data_train = llm4_preprocess(l4in, l4outputs)
llm4_preprocessed_data_test = llm4_preprocess(l4Vin, l4Voutputs)


# Global Parameters
L_RATE = 1e-5
BATCH_SIZE = 8
PER_DEVICE_EVAL_BATCH = 4
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 5
NUM_EPOCHS = 120

training_args = tf.Seq2SeqTrainingArguments(
    output_dir="./llm_tuning",
    evaluation_strategy="epoch",
    learning_rate=L_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
    weight_decay=WEIGHT_DECAY,
    save_strategy="epoch",
    save_total_limit=SAVE_TOTAL_LIM,
    num_train_epochs=NUM_EPOCHS,
    predict_with_generate=True,
    push_to_hub=False,
    load_best_model_at_end=True
)
llm2_trainer = tf.Seq2SeqTrainer(
    model=llm2_model,
    args=training_args,
    train_dataset=preprocessed_data_train,
    eval_dataset=preprocessed_data_test,
    tokenizer=tokenizer,
    data_collator=llm2_collator,
    compute_metrics=evaluate_metrics
)
llm4_trainer = tf.Seq2SeqTrainer(
    model=llm4_model,
    args=training_args,
    train_dataset=llm4_preprocessed_data_train,
    eval_dataset=llm4_preprocessed_data_test,
    tokenizer=tokenizer,
    data_collator=llm4_collator,
    compute_metrics=evaluate_metrics
)

#the pllm4_trainer.train()

