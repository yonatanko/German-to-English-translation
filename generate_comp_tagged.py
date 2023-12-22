from tqdm import tqdm
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

"""uploading the model"""
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained("our_dir/checkpoint-10000")
model = AutoModelForSeq2SeqLM.from_pretrained("our_dir/checkpoint-10000").to(device)


def convert_to_tuple_list(input_str):
    input_list = input_str.split('), ')

    for i in range(len(input_list)):
        open_paren_count = input_list[i].count('(')
        close_paren_count = input_list[i].count(')')

        if open_paren_count > close_paren_count:
            input_list[i] += ')' * (open_paren_count - close_paren_count)

    return input_list


def create_new_val():
    flag = False
    with open('val.unlabeled', 'r', encoding='utf-8') as first, open('new_val.labeled', 'w', encoding='utf-8') as new:
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


def read_file(file_path):
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
            cur_str += line + ' '
    if len(cur_str) > 0:
        cur_list.append(cur_str)
    return file_en, file_de


def read_file_with_DP(file_path):
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


def tagger(test_file_de, test_roots, test_modifiers, file):
    for i in tqdm(range(len(test_file_de))):
        text = test_file_de[i]
        file.write('German: ')
        file.write('\n')
        file.write(text)
        file.write('\n')
        file.write('English: ')
        file.write('\n')
        p_roots = test_roots[i]
        p_modifiers = test_modifiers[i]
        template = f"For the following German paragraph {text}, translate each sentence with the corresponding English root and modifiers: "
        for root, modifiers in zip(p_roots, p_modifiers):
            template += root + " " + modifiers + ", "
        final_input = template[:-2]
        inputs = tokenizer(final_input, return_tensors="pt", truncation=True).input_ids
        output = model.generate(inputs.to(device), do_sample=True, max_length=512, top_k=5, top_p=0.8, num_beams=2)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        file.write(output)
        file.write('\n')
        file.write('\n')


def main():
    create_new_val()
    _, test_file_de, test_roots, test_modifiers = read_file_with_DP('new_val.labeled')
    _, comp_file_de, comp_roots, comp_modifiers = read_file_with_DP('comp.unlabeled')

    val_with_preds = open('val_212984801_207261983.labeled', 'w', encoding='utf-8')
    comp_with_preds = open('comp_212984801_207261983.labeled', 'w', encoding='utf-8')
    tagger(test_file_de, test_roots, test_modifiers, val_with_preds)
    tagger(comp_file_de, comp_roots, comp_modifiers, comp_with_preds)

if __name__ == "__main__":
    main()
