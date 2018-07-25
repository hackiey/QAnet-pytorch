import os
import json
import random
import numpy as np
import spacy
import pickle

# from
from collections import Counter
from tqdm import tqdm

nlp = spacy.blank("en")


def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print("Token {} cannot be found".format(token))
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print("Generating {} examples...".format(data_type))
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                # tokenize
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    # tokenize
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens, "context_chars": context_chars, "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars, "y1s": y1s, "y2s": y2s, "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {
                        "context": context, "spans": spans, "answers": answer_texts, "uuid": qa["id"]}
        print("{} questions in total".format(len(examples)))
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, size=None, vec_size=None):
    print("Generating {} embedding...".format(data_type))
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert size is not None
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=size):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print("{} / {} tokens have corresponding {} embedding vector".format(
            len(embedding_dict), len(filtered_elements), data_type))
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print("{} tokens have corresponding embedding vector".format(
            len(filtered_elements)))

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx,
                      token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def build_features(params, examples, data_type, word2idx_dict, char2idx_dict):

    para_limit = params['para_limit']
    ques_limit = params['ques_limit']
    ans_limit = params['ans_limit']
    char_limit = params['char_limit']

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit or \
               (example["y2s"][0] - example["y1s"][0]) > ans_limit

    print("Processing {} examples...".format(data_type))
    # writer = tf.python_io.TFRecordWriter(out_file)
    total = 0
    total_ = 0
    meta = {}

    context_idxs_list = []
    ques_idxs_list = []
    context_char_idxs_list = []
    ques_char_idxs_list = []
    y_list = []
    ids_list = []

    for example in tqdm(examples):
        total_ += 1

        if filter_func(example):
            continue

        total += 1
        context_idxs = np.zeros([para_limit], dtype=np.int32)
        context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idxs = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)
        y1 = np.zeros([para_limit], dtype=np.float32)
        y2 = np.zeros([para_limit], dtype=np.float32)

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        for i, token in enumerate(example["context_tokens"]):
            context_idxs[i] = _get_word(token)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idxs[i] = _get_word(token)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idxs[i, j] = _get_char(char)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idxs[i, j] = _get_char(char)

        start, end = example["y1s"][-1], example["y2s"][-1]
        y1[start], y2[end] = 1.0, 1.0

        context_idxs_list.append(context_idxs)
        ques_idxs_list.append(ques_idxs)
        context_char_idxs_list.append(context_char_idxs)
        ques_char_idxs_list.append(ques_char_idxs)
        y_list.append([start, end])
        ids_list.append(example['id'])

    print("Built {} / {} instances of features in total".format(total, total_))
    meta["total"] = total

    pickle.dump(context_idxs_list, open(os.path.join(params['target_dir'], data_type + '_context_idxs.pkl'), 'wb'))
    pickle.dump(ques_idxs_list, open(os.path.join(params['target_dir'], data_type + '_ques_idxs.pkl'), 'wb'))
    pickle.dump(context_char_idxs_list, open(os.path.join(params['target_dir'], data_type + '_context_char_idxs.pkl'), 'wb'))
    pickle.dump(ques_char_idxs_list, open(os.path.join(params['target_dir'], data_type + '_ques_char_idxs.pkl'), 'wb'))
    pickle.dump(y_list, open(os.path.join(params['target_dir'], data_type + '_y.pkl'), 'wb'))
    pickle.dump(ids_list, open(os.path.join(params['target_dir'], data_type+'_ids.pkl'), 'wb'))

    return meta


def preprocess(params):
    # files
    train_file = os.path.join(params['data_dir'], params['train_file'])
    dev_file = os.path.join(params['data_dir'], params['dev_file'])
    test_file = os.path.join(params['data_dir'], params['test_file'])
    word_emb_file = os.path.join(params['glove_dir'], params['word_embedding_file'])

    word_counter, char_counter = Counter(), Counter()

    # examples
    train_examples, train_eval = process_file(train_file, "train", word_counter, char_counter)
    dev_examples, dev_eval = process_file(dev_file, "dev", word_counter, char_counter)
    test_examples, test_eval = process_file(test_file, 'test', word_counter, char_counter)

    # embedding
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, "word", emb_file=word_emb_file, size=params['glove_word_size'], vec_size=params['glove_dim'])
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, "char", emb_file=None, size=None, vec_size=params['char_dim'])

    pickle.dump(train_examples, open(os.path.join(params['target_dir'], 'train_examples.pkl'), 'wb'))
    pickle.dump(train_eval, open(os.path.join(params['target_dir'], 'train_eval.pkl'), 'wb'))
    pickle.dump(dev_examples, open(os.path.join(params['target_dir'], 'dev_examples.pkl'), 'wb'))
    pickle.dump(dev_eval, open(os.path.join(params['target_dir'], 'dev_eval.pkl'), 'wb'))
    pickle.dump(test_examples, open(os.path.join(params['target_dir'], 'test_examples.pkl'), 'wb'))
    pickle.dump(test_eval, open(os.path.join(params['target_dir'], 'test_eval.pkl'), 'wb'))

    pickle.dump(word_emb_mat, open(os.path.join(params['target_dir'], 'word_emb_mat.pkl'), 'wb'))
    pickle.dump(word2idx_dict, open(os.path.join(params['target_dir'], 'word2idx_dict.pkl'), 'wb'))
    pickle.dump(char_emb_mat, open(os.path.join(params['target_dir'], 'char_emb_mat.pkl'), 'wb'))
    pickle.dump(char2idx_dict, open(os.path.join(params['target_dir'], 'char2idx_dict.pkl'), 'wb'))

    pickle.dump(word_counter, open(os.path.join(params['target_dir'], 'word_counter.pkl'), 'wb'))
    pickle.dump(char_counter, open(os.path.join(params['target_dir'], 'char_counter.pkl'), 'wb'))

    # ======================== need remove ===============================
    # train_examples = json.load(open('data/train_examples.json', 'r'))
    # train_eval = json.load(open('data/train_eval.json', 'r'))
    # dev_examples = json.load(open('data/dev_examples.json', 'r'))
    # dev_eval = json.load(open('data/dev_eval.json', 'r'))

    # word_counter = pickle.load(open('tmp/word_counter.pkl', 'rb'))
    # char_counter = pickle.load(open('tmp/char_counter.pkl', 'rb'))

    # word_emb_mat = pickle.load(open('data/word_emb_mat.pkl', 'rb'))
    # word2idx_dict = pickle.load(open('data/word2idx_dict.pkl', 'rb'))

    # char_emb_mat = pickle.load(open('data/char_emb_mat.pkl', 'rb'))
    # char2idx_dict = pickle.load(open('data/char2idx_dict.pkl', 'rb'))
    # ====================================================================

    build_features(params, train_examples, "train", word2idx_dict, char2idx_dict)
    dev_meta = build_features(params, dev_examples, "dev", word2idx_dict, char2idx_dict)
    test_meta = build_features(params, test_examples, "test", word2idx_dict, char2idx_dict)

if __name__ == '__main__':
    params = json.load(open('params.json', 'r'))
    preprocess(params)
    print('yeah')
