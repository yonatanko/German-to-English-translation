import spacy.cli
import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, \
    AutoTokenizer
import logging
from tqdm import tqdm
import torch
import spacy
import string
import pandas as pd
import random

# Download the "en_core_web_sm" model
spacy.cli.download("en_core_web_sm")
punctuations = string.punctuation
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelForSeq2SeqLM.from_pretrained("t5-base").to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
metric = evaluate.load("sacrebleu")


"""Adding roots and modifiers"""


def read_file_for_DP(file_path):
    file_en, file_de = [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue
            if line != "":
                cur_str += line + '<Seperator>'

    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def separate_to_sentences(en_paragraphs, de_paragraphs):
    english_sen, de_sen = [], []

    for p1, p2 in zip(en_paragraphs, de_paragraphs):
        curr_sen = p1.split("<Seperator>")[:-1]
        english_sen += curr_sen
        english_sen += ["end_of_paragraph"]

        curr_sen = p2.split("<Seperator>")[:-1]
        de_sen += curr_sen
        de_sen += ["end_of_paragraph"]

    return english_sen, de_sen


def remove_punct(modifiers):
    new_list = []
    for element in modifiers:
        elem = str(element)
        if elem not in punctuations:
            new_list.append(elem)
    return new_list


def find_roots_and_modifiers(en_sen):
    all_p_roots = []
    all_p_modifiers = []
    curr_p_roots = []
    curr_p_modifiers = []

    for i in tqdm(range(len(en_sen))):
        sen = en_sen[i]
        if sen != "end_of_paragraph":
            doc = nlp(sen)
            for token in doc:
                if token.dep_ == "ROOT":
                    curr_p_roots.append(token.text)
                    modifiers = [child for child in token.children]
                    modifiers = remove_punct(modifiers)
                    curr_p_modifiers.append(tuple(modifiers))
            if en_sen[i + 1] == "end_of_paragraph":
                all_p_roots.append(curr_p_roots)
                all_p_modifiers.append(curr_p_modifiers)
                curr_p_roots = []
                curr_p_modifiers = []

    return all_p_roots, all_p_modifiers


def generate_file_with_DP(from_file, to_file, all_p_roots, all_p_modifiers):
    paragraph_index = 0

    with open(from_file, 'r', encoding='utf-8') as original, open(to_file, 'w', encoding='utf-8') as withDP:
        for line in original.readlines():
            if line != "\n":
                withDP.write(line)
            else:
                new_line_roots = ""
                new_line_roots += "Roots in English: "
                for root in all_p_roots[paragraph_index]:
                    new_line_roots += root + ', '

                new_line_modifiers = "Modifiers in English: "
                for modifiers in all_p_modifiers[paragraph_index]:
                    new_line_modifiers += '(' + ', '.join(modifiers) + ')' + ', '

                new_line_modifiers = new_line_modifiers[:-2]
                new_line_roots = new_line_roots[:-2]

                withDP.write(new_line_roots)
                withDP.write("\n")
                withDP.write(new_line_modifiers)
                withDP.write("\n")
                withDP.write("\n")
                paragraph_index += 1


def create_new_val():  # Creating val file that will have all the information in one place (German + english + roots and modifiers)
    flag = False
    with open('val.unlabeled', 'r', encoding='utf-8') as first, open('new_val.labeled', 'a', encoding='utf-8') as new:
        data = first.readlines()
        for line in data:
            new.write(line)
    with open('new_val.labeled', 'a', encoding='utf-8') as new, open('val.labeled', 'r', encoding='utf-8') as second:
        unlabeled = second.readlines()
        for line in unlabeled:
            if 'English:' in line:
                flag = True
            if 'German:' in line:
                flag = False
            if flag:
                new.write(line)


def create_train_with_only_2_modifiers(first, second):
    with open(first, 'r', encoding='utf-8') as f, open(second, 'w', encoding='utf-8') as new:
        data = f.readlines()
        for line in data:
            if 'Modifiers' in line:
                new_line = line[:22]
                cur_line = line[22:]
                cur_line = cur_line.replace('\n', '')
                cur_line = cur_line.replace('),', ' ')
                cur_line = cur_line.replace(')', '')
                cur_line = cur_line.replace('(', ' ')
                cur_line = cur_line.split('  ')
                for i in cur_line:
                    i = i.split(',')
                    if len(i) > 1:
                        random.shuffle(i)
                        new_line += '(' + str(i[0][1:]) + ', ' + str(i[1][1:]) + '), '
                    else:
                        new_line += '(' + str(i[0][1:]) + '), '

                new.write(new_line[:-2])
            else:
                new.write(line)


"""Extracting the data"""


def convert_to_tuple_list(input_str):
    input_list = input_str.split('), ')

    for i in range(len(input_list)):
        open_paren_count = input_list[i].count('(')
        close_paren_count = input_list[i].count(')')

        if open_paren_count > close_paren_count:
            input_list[i] += ')' * (open_paren_count - close_paren_count)

    return input_list


def read_file_with_DP(file_path):  # Reading file that has roots and modifiers for each paragraph
    file_en, file_de, all_p_roots, all_p_modifiers = [], [], [], []
    with open(file_path, encoding='utf-8') as f:
        cur_str, cur_list = '', []
        for line in f.readlines():
            line = line.strip()
            if line == 'English:' or line == 'German:':
                if len(cur_str) > 0:
                    cur_list.append(cur_str.strip())
                    cur_str = ''
                if line == 'English:':
                    cur_list = file_en
                else:
                    cur_list = file_de
                continue

            if "Roots in English" in line:
                line = line[line.index(':') + 2:]
                curr_roots = line.split(', ')

            elif "Modifiers in English" in line:
                line = line[line.index(':') + 2:]
                curr_modifiers = convert_to_tuple_list(line)

                all_p_roots.append(curr_roots)
                all_p_modifiers.append(curr_modifiers)

            else:
                cur_str += line + ' '

    if len(cur_str) > 0:
        cur_list.append(cur_str)

    return file_en, file_de, all_p_roots, all_p_modifiers


def to_csv(file_en, file_de, data, name, all_p_roots, all_p_modifiers):
    for en_p, de_p, p_roots, p_modifiers in zip(file_en, file_de, all_p_roots, all_p_modifiers):
        template = f"For the following German paragraph: {de_p}, translate each sentence with the corresponding English root and modifiers: "
        for root, modifiers in zip(p_roots, p_modifiers):
            template += root + " " + modifiers + ", "
        final_input = template[:-2]
        data.append([final_input, en_p])

    data_df = pd.DataFrame(data, columns=["text", "labels"])
    data_df.to_csv(f"{name}.csv", index=False)


def preprocess_function(examples):
    inputs = [example for example in examples["text"]]
    targets = [example for example in examples["labels"]]
    model_inputs = tokenizer(inputs, truncation=True, max_length=200)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, truncation=True, max_length=200)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def compute_metrics_BLEU(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # Replace -100 in the labels as we can't decode them.
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    result = {"bleu": result["score"]}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result


def compute_metrics(tagged_en, true_en):
    tagged_en = [x.strip().lower() for x in tagged_en]
    true_en = [x.strip().lower() for x in true_en]

    result = metric.compute(predictions=tagged_en, references=true_en)
    result = result['score']
    result = round(result, 2)
    return result


def tagger(test_file, test_roots, test_modifiers, new_tokenizer, new_model):
    preds = []
    for i in tqdm(range(len(test_file))):
        text = test_file[i]
        p_roots = test_roots[i]
        p_modifiers = test_modifiers[i]
        template = f"For the following German paragraph {text}, translate each sentence with the corresponding English root and modifiers: "
        for root, modifiers in zip(p_roots, p_modifiers):
            template += root + " " + modifiers + ", "
        final_input = template[:-2]
        inputs = new_tokenizer(final_input, return_tensors="pt", truncation=True).input_ids
        output = new_model.generate(inputs.to(device), do_sample=True, max_length=512, top_k=10, top_p=0.8, num_beams=2)
        output = new_tokenizer.decode(output[0], skip_special_tokens=True)
        preds.append(output)
    return preds


def main():
    # Reading initial train data and calculating roots and modifiers
    train_en_paragraphs, train_de_paragraphs = read_file_for_DP('train.labeled')
    train_en_sen, train_de_sen = separate_to_sentences(train_en_paragraphs, train_de_paragraphs)
    train_roots, train_modifiers = find_roots_and_modifiers(train_en_sen)

    # Creating new train files and new eval file
    generate_file_with_DP('train.labeled', 'train_with_DP.labeled', train_roots, train_modifiers)
    create_train_with_only_2_modifiers('train_with_DP.labeled', 'train_with_2.labeled')
    create_new_val()

    # Reading new data and creating datasets
    train_file_en, train_file_de, train_roots, train_modifiers = read_file_with_DP('train_with_2.labeled')
    test_file_en, test_file_de, test_roots, test_modifiers = read_file_with_DP('new_val.labeled')
    to_csv(train_file_en, train_file_de, [], 'train', train_roots, train_modifiers)
    to_csv(test_file_en, test_file_de, [], 'test', test_roots, test_modifiers)
    data_files = {
        'train': 'train.csv',
        'test': 'test.csv'
    }
    raw_datasets = load_dataset("csv", data_files=data_files)

    # Tokenizing the datasets
    tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
    tokenized_datasets.set_format('torch')

    # Defining training arguments and starting to fine-tune the t5-base model
    training_args = Seq2SeqTrainingArguments(
        output_dir="our_dir",
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=4,
        save_total_limit=3,
        predict_with_generate=True,
        generation_max_length=512,
        generation_num_beams=2,
        optim="adamw_torch",
        save_strategy="epoch"
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics_BLEU
    )

    trainer.train()

    new_tokenizer = AutoTokenizer.from_pretrained("our_dir/checkpoint-10000")
    new_model = AutoModelForSeq2SeqLM.from_pretrained("our_dir/checkpoint-10000").to(device)

    logging.getLogger("transformers").setLevel(logging.ERROR)
    preds = tagger(test_file_de, test_roots, test_modifiers, new_tokenizer, new_model)
    result = compute_metrics(test_file_en, preds)
    print(f"BLEU on val: ")
    print(result)


if __name__ == "__main__":
    main()
