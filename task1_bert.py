from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer
from transformers import default_data_collator
from tqdm import tqdm
import torch
import utils

train = True
model_checkpoint = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)

def preprocess_function(examples):
    short_questions = [q.split('\t')[0][5:] for q in examples["question"]]

    questions = [q.strip() for q in short_questions]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        # print(start_char, end_char, sequence_ids)
        # print(offset)
        # print()

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

def preprocess_val(examples):
    short_questions = [q.split('\t')[0][5:] for q in examples["question"]]

    questions = [q.strip() for q in short_questions]

    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=512,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

#     offset_mapping = inputs.pop("offset_mapping")
    offset_mapping = inputs["offset_mapping"]
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)
        # print(start_char, end_char, sequence_ids)
        # print(offset)
        # print()

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    inputs["answers"] = examples["answers"]
    inputs["question"] = questions
    inputs["context"] = examples["context"]
    return inputs


if __name__ == '__main__':
    # get data
    train_data = utils.load_doc2dial_dataset(name="doc2dial_rc", split="train")
    val_data = utils.load_doc2dial_dataset(name="doc2dial_rc", split="validation")
    # tokenise data
    tok_train_data = train_data.map(preprocess_function, batched=True, remove_columns=train_data.column_names)
    tok_val_data = val_data.map(preprocess_function, batched=True, remove_columns=val_data.column_names)


    # train
    if train:
        batch_size = 50
        args = TrainingArguments(
            f"{model_checkpoint}-finetuned-doc2dial",
        #     evaluation_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=2,
            weight_decay=0.01,
            evaluation_strategy="steps",
        #     logging_strategy="steps",
            logging_steps=100
        )

        trainer = Trainer(
            model,
            args,
            train_dataset=tok_train_data,
            eval_dataset=tok_val_data,
            data_collator=default_data_collator,
            tokenizer=tokenizer
        )
    else:
        checkpoint = "distilbert-base-uncased-finetuned-doc2dial/checkpoint-1000/"
        checkpoint = "bert_matt_1/"
        trainer = AutoModelForQuestionAnswering.from_pretrained(checkpoint)
        tok_val_data = val_data.map(preprocess_val, batched=True)

## Evaluate
softmax = torch.nn.functional.softmax

def format_answer(qid, text, proba):
    return {"id":qid, "prediction_text":text, "no_answer_probability":1-proba}

# loop over to get all predictions
all_preds = []
for i in tqdm(range(0, tok_val_data.num_rows)):
    example = tok_val_data.select([i])
    short_question = [q.split('\t')[0][5:] for q in example["question"]]
    question = [q.strip() for q in short_question]

    inputs = tokenizer(question,
                       example["context"],
                        max_length=512,
                        truncation="only_second",
                        return_offsets_mapping=True,
                        padding="max_length",
                        return_tensors="pt"
        )

    offsets = inputs["offset_mapping"]
    sequence_ids = inputs.sequence_ids()

    # inputs = tok_val_data.remove_columns(['answers', 'context', 'domain', 'id', 'end_positions',  'question', 'start_positions', 'title'])
    # inputs

    # outputs = trainer(**inputs)
    outputs = trainer(inputs["input_ids"], inputs["attention_mask"])

    context = example[0]["context"]
    qid = example[0]["id"]
    offsets = example[0]["offset_mapping"]

    start_logits = softmax(outputs.start_logits[0], dim=0).detach().numpy()
    end_logits = softmax(outputs.end_logits[0], dim=0).detach().numpy()

#     start_logits = softmax(outputs.start_logits[i].detach().numpy())
#     end_logits = softmax(outputs.end_logits[i].detach().numpy())

    start_index = np.argsort(start_logits)[-1]
    end_index = np.argsort(end_logits)[-1]


    ## Checks the answer
    score = (start_logits[start_index] + end_logits[end_index]) / 2.0
#     print(start_logits[start_index], end_logits[end_index])

    if start_index >= end_index:
#         print("No answer")
        all_preds.append(format_answer(qid, "", score))
    elif sequence_ids[start_index] == 0 or sequence_ids[start_index] ==None:
#         print('Invalid answer')
        all_preds.append(format_answer(qid, "", score))
    else:
        start_char = offsets[start_index][0]
        end_char = offsets[end_index][1]
        text = context[start_char: end_char]
#         print("starting en ending positions", start_pos, end_pos)
        all_preds.append(format_answer(qid, text, score))
