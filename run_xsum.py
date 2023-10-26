from collections import defaultdict
import pickle
import spacy
import torch
import argparse
import sys
from sklearn.metrics import precision_recall_curve, roc_auc_score, classification_report, balanced_accuracy_score, auc
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from models.modeling_llama import LlamaForCausalLM
from tqdm import tqdm
import numpy as np

out = []
nlp = spacy.load('en_core_web_sm')
out = []
left_mark = '<'
right_mark = '>'
thres_num = 11

def generate_summac():
    from data.summac import SummaCBenchmark
    benchmark_val = SummaCBenchmark(benchmark_folder="data/summac_benchmark/", cut="test")
    return benchmark_val.datasets

def run_generate(prompt, use_threshold=False, discount=0., rm_low_prob=False, use_idf=False):
    NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
    pos_tag = ["NOUN", "NUM", "PROPN"]
    start_token_idx = 999999
    with torch.no_grad():
        encodings = tokenizer(prompt, return_tensors="pt")
        if encodings["input_ids"].size(1) > 2048:
            return None, None
        for i in range(len(prompt)):
            if prompt[max(i - 6, 0):i] == '\nTL;DR':
                start_token_idx = min(start_token_idx, encodings.char_to_token(i))
                break
        input_ids = encodings.input_ids

        assert len(input_ids) == 1, len(input_ids)

        input_ids = input_ids.to(0)

        outputs = model(input_ids=input_ids, labels=input_ids, output_attentions=True, rm_low_prob=rm_low_prob,
                        use_idf=use_idf)
        loss = outputs["loss"][0]
        attention = outputs["attentions"][:-1, :-1]
        attention[:start_token_idx, :] = 0
        attention[:, :start_token_idx] = 0
        attention.fill_diagonal_(0)
        noun_mask = torch.ones_like(loss)
        if use_threshold:
            loss = loss.view(-1, 1).repeat(1, thres_num)

        words = []
        losses = []
        for sentence in nlp(prompt).sents:
            for span_idx, span in enumerate(sentence):
                start, end = span.idx, span.idx + len(span)
                cur_word = prompt[start:end]
                if len(cur_word.replace('\n', '')) == 0 and encodings.char_to_token(start) > start_token_idx:
                    continue
                words.append(cur_word)
                num = 0
                if use_threshold:
                    l = [0] * thres_num
                else:
                    l = 0
                memo = {}
                for i in range(start, end):
                    loss_index = encodings.char_to_token(i) - 1
                    if loss_index not in memo:
                        if span.text not in NER_type and (span.ent_type_ in NER_type or span.pos_ in pos_tag):
                            noun_mask[loss_index] = 0
                            attention[loss_index][noun_mask.bool()] = 0
                            weight = attention[loss_index] / (torch.sum(attention[loss_index]) + 1e-6)
                            if use_threshold:
                                weight = weight.view(-1, 1)
                                penalty = torch.sum(weight * loss, dim=0).tolist()
                                for threshold in range(thres_num):
                                    loss[loss_index][threshold] += threshold / 10 * penalty[threshold]
                            else:
                                penalty = torch.sum(weight * loss).item()
                                loss[loss_index] += discount * penalty
                        if use_threshold:
                            for threshold in range(thres_num):
                                l[threshold] += loss[loss_index][threshold].item() if loss_index >= 0 else 0
                        else:
                            l += loss[loss_index].item() if loss_index >= 0 else 0
                        num += 1
                        memo[loss_index] = True
                if use_threshold:
                    losses.append([l_thres/num for l_thres in l])
                else:
                    losses.append(l / num)
                if words[-1] == right_mark and (words[-2] in NER_type + pos_tag) and words[-3] == left_mark:
                    words = words[:-3]
                    losses = losses[:-3]
    return words, losses

def evaluate(loss_data, pooling="mean", top=5, only_noun=True):
    NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
    pos_tag = ["NOUN", "NUM", "PROPN"]
    labels = [label[i] for i, d in enumerate(loss_data[0]) if d is not None]
    summaries = [summac_dataset[i]["claim"] for i, d in enumerate(loss_data[0]) if d is not None]
    loss_data = ([d for d in loss_data[0] if d is not None], [d for d in loss_data[1] if d is not None])
    if isinstance(loss_data[1][0][0], list):
        loss_threshold = [[[loss_word[thres_num] for loss_word in example] for example in loss_data[1]] for thres_num in
                          range(len(loss_data[1][0][0]))]
    else:
        loss_threshold = [loss_data[1]]
    for loss_thres_idx in range(len(loss_threshold)):
        source = []
        target = []
        summary_list = []
        article_list = []
        _words = []
        _loss = []
        res = defaultdict(list)
        for i in range(len(loss_data[0])):
            total_words = []
            for j, word in enumerate(loss_data[0][i]):
                if word == 'TL;DR' and '\n' in loss_data[0][i][j - 1]:
                    begin_index = j + 2
                    break
            loss = loss_threshold[loss_thres_idx][i][begin_index:]
            words = loss_data[0][i][begin_index:]
            summary = summaries[i].strip()
            cur = 0
            summary_loss_noun = 0 if pooling == "mean" else []
            summary_loss_not_noun = 0
            summary_loss = 0
            num_noun = 0
            num_not_noun = 0
            num = 0
            tmp_word = []
            tmp_loss = []
            words_list = [t for t in nlp(summary) if len(t.text.replace('\n', '')) != 0]
            for w in words_list:
                if w.ent_type_ in NER_type or w.pos_ in pos_tag:
                    if cur == len(loss):
                        print(1)
                    summary_loss_noun += loss[cur] if pooling == "mean" else [loss[cur]]
                    num_noun += 1
                    tmp_word.append(w)
                    tmp_loss.append(loss[cur])
                else:
                    summary_loss_not_noun += loss[cur]
                    num_not_noun += 1
                summary_loss += loss[cur]
                num += 1
                cur += 1
                total_words.append(w)
            if pooling == "mean":
                res[f"{'hallucination' if labels[i] == 1 else 'non_hallucination'}_noun"].append(
                    (summary_loss_noun / (num_noun + 0.001)))
            else:
                sorted_loss = sorted(summary_loss_noun, reverse=True)[:top]
                res[f"{'hallucination' if labels[i] == 1 else 'non_hallucination'}_noun"].append(
                    sum(sorted_loss) / len(sorted_loss))
            res[f"{'hallucination' if labels[i] == 1 else 'non_hallucination'}_not_noun"].append(
                summary_loss_not_noun / (num_not_noun + 0.001))
            res[f"{'hallucination' if labels[i] == 1 else 'non_hallucination'}"].append(summary_loss / num)
            source.append(
                res[f"{'hallucination' if labels[i] == 1 else 'non_hallucination'}_noun"][-1] if only_noun else
                res[f"{'hallucination' if labels[i] == 1 else 'non_hallucination'}"][-1])
            target.append(labels[i])
            summary_list.append(summary)
            article_list.append(summac_dataset[i]['document'])
            _words.append(tmp_word)
            _loss.append(tmp_loss)
            if len(words) != cur:
                print(1)
        print(f"non_factual: {calcu_auc_pr(source, target)} || factual: {calcu_auc_pr_reverse(source, target)}")
        max_acc = -1
        for i in source:
            predictions = [1 if s > i else 0 for s in source]
            max_acc = max(balanced_accuracy_score(target, predictions), max_acc)
        print("acc:", max_acc)
        if not only_noun:
            print("---------------------")
            break
    return res

def add_type(text):
    NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT",
                "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
    words = [t for t in nlp(text)]
    word_type = [t.ent_type_ if t.ent_type_ in NER_type else '' for t in words]
    cur_type = word_type[0]
    cur = 1
    while cur < len(word_type):
        if word_type[cur] == cur_type:
            word_type[cur] = ''
        else:
            cur_type = word_type[cur]
        cur += 1
    offset = 0
    for i, w in enumerate(words):
        start = w.idx + offset
        text = f"{text[:start]}{left_mark}{word_type[i]}{right_mark} {text[start:]}" if len(word_type[i]) and (
                text[start - 1] == ' ' or start == 0) else text
        offset += len(f"{left_mark}{word_type[i]}{right_mark} ") if len(word_type[i]) and (
                text[start - 1] == ' ' or start == 0) else 0
    return text


def calcu_auc_pr(source, target, draw=False):
    labels = [1 if t > 0 else 0 for t in target]
    predictions = [s / max(source) for s in source]
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    auc_precision_recall = auc(recall, precision)
    if draw:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        writer.add_pr_curve('pr_curve_non_factual', np.array(labels), np.array(predictions))
        writer.close()
    return auc_precision_recall


def calcu_auc_pr_reverse(source, target, draw=False):
    labels = [0 if t > 0 else 1 for t in target]
    predictions = [1 - s / max(source) for s in source]
    precision, recall, thresholds = precision_recall_curve(labels, predictions)
    auc_precision_recall = auc(recall, precision)
    if draw:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter()
        writer.add_pr_curve('pr_curve_non_factual', np.array(labels), np.array(predictions))
        writer.close()
    return auc_precision_recall


def filter_dataset(dataset):
    if dataset["name"] == 'xsumfaith':
        not_remain_type = ['Gold']
        filtered_list = [d for d in dataset["dataset"] if d['model_name'] not in not_remain_type]
    dataset["dataset"] = filtered_list
    return [dataset]


datasets = generate_summac()
datasets = filter_dataset(datasets[1])


def calcu_auc(source, target):
    labels = target
    predictions = [s / max(source) for s in source]
    print(roc_auc_score(labels, predictions))
    print(classification_report(labels, predictions))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/weights/llama/hf/")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    args = parser.parse_args()

    # "ausboss/llama-30b-supercot"
    model_paths = [args.model_path]
    for model_path in model_paths:
        tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/huggingface_models/llama-65b", use_fast=True)
        model = LlamaForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            cache_dir="/home/ubuntu/huggingface_models"
        )

        for summac_dataset in datasets:
            print(summac_dataset["name"])
            text = [f"{d['document']}\nTL;DR\n{d['claim']}" for d in summac_dataset["dataset"]]
            label = [1 - d['label'] for d in summac_dataset["dataset"]]
            words_list, losses_list = [], []
            for i in tqdm(range(len(text))):
                words, losses = run_generate(text[i], use_threshold=True, rm_low_prob=False)
                words_list.append(words)
                losses_list.append(losses)
            print(model_path.split('/')[-1])
            name = summac_dataset['name']
            summac_dataset = summac_dataset["dataset"]
            evaluate((words_list, losses_list), only_noun=False)
            evaluate((words_list, losses_list), only_noun=True)

        for summac_dataset in datasets:
            print(summac_dataset["name"])
            text = [
                f"Summarize the following text using appropriate words that follow to the given type:\n{add_type(d['document'])}\nTL;DR\n{add_type(d['claim'])}"
                for d in summac_dataset["dataset"]]
            label = [1 - d['label'] for d in summac_dataset["dataset"]]
            name = summac_dataset['name']
            summac_dataset = summac_dataset["dataset"]

            def test(p, use_idf=False):
                words_list, losses_list = [], []
                for i in tqdm(range(len(p))):
                    words, losses = run_generate(p[i], use_threshold=True, rm_low_prob=True, use_idf=use_idf)
                    words_list.append(words)
                    losses_list.append(losses)
                print(model_path.split('/')[-1])
                evaluate((words_list, losses_list))

            test(text, use_idf=True)
            print("------------no idf------------")
            test(text, use_idf=False)
