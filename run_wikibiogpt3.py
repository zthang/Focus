import pickle
from collections import defaultdict
import spacy
import torch
import argparse
import sys
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizerFast
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc
from models.modeling_opt import OPTForCausalLM
from models.modeling_llama import LlamaForCausalLM
from models.modeling_gpt_neox import GPTNeoXForCausalLM
from models.modeling_gptj import GPTJForCausalLM
from tqdm import tqdm
import numpy as np
import scipy

out = []
left_mark = '<'
right_mark = '>'
nlp = spacy.load('en_core_web_sm')

def run_generate(prompt, use_threshold=False, discount=0., rm_low_prob=False, use_idf=False):
    NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT", "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
    pos_tag = ["NOUN", "NUM", "PROPN"]
    start_token_idx = 999999
    with torch.no_grad():
        encodings = tokenizer(prompt, return_tensors="pt")
        for i in range(len(prompt)):
            if prompt[i-1:i+1] == ':\n':
                start_token_idx = min(start_token_idx, encodings.char_to_token(i))
                break
        input_ids = encodings.input_ids

        assert len(input_ids) == 1, len(input_ids)
        input_ids = input_ids.to(0)

        outputs = model(input_ids=input_ids, labels=input_ids, output_attentions=True, rm_low_prob=rm_low_prob, use_idf=use_idf)
        loss = outputs["loss"][0]
        attention = outputs["attentions"][:-1, :-1]
        attention[:start_token_idx, :] = 0
        attention[:, :start_token_idx] = 0
        attention.fill_diagonal_(0)
        noun_mask = torch.ones_like(loss)
        if use_threshold:
            loss = loss.view(-1, 1).repeat(1, 11)
        words = []
        losses = []
        nums = []
        for sentence in nlp(prompt).sents:
            for span_idx, span in enumerate(sentence):
                start, end = span.idx, span.idx+len(span)
                cur_word = prompt[start:end]
                if len(cur_word.replace('\n', '')) == 0 and encodings.char_to_token(start) > start_token_idx:
                    continue
                words.append(cur_word)
                num = 0
                if use_threshold:
                    l = [0] * 11
                else:
                    l = 0
                memo = {}
                for i in range(start, end):
                    loss_index = encodings.char_to_token(i)-1
                    if loss_index not in memo:
                        if span.text not in NER_type and (span.ent_type_ in NER_type or span.pos_ in pos_tag):
                            noun_mask[loss_index] = 0
                            attention[loss_index][noun_mask.bool()] = 0
                            weight = attention[loss_index]/(torch.sum(attention[loss_index])+1e-6)
                            if use_threshold:
                                weight = weight.view(-1, 1)
                                penalty = torch.sum(weight * loss, dim=0).tolist()
                                for threshold in range(11):
                                    loss[loss_index][threshold] += threshold / 10 * penalty[threshold]
                            else:
                                penalty = torch.sum(weight * loss).item()
                                loss[loss_index] += discount * penalty
                        if use_threshold:
                            for threshold in range(11):
                                l[threshold] += loss[loss_index][threshold].item() if loss_index >= 0 else 0
                        else:
                            l += loss[loss_index].item() if loss_index >= 0 else 0
                        num += 1
                        memo[loss_index] = True
                if use_threshold:
                    losses.append([l_thres for l_thres in l])
                    nums.append(num)
                else:
                    losses.append(l / num)
                if words[-1] == right_mark and (words[-2] in NER_type + pos_tag) and words[-3] == left_mark:
                    words = words[:-3]
                    losses = losses[:-3]
                    nums = nums[:-3]
        return words, losses, nums

def evaluate(loss_data, pooling="mean", top=5, only_noun=True):
    non_factual_list = []
    non_factual_star_list = []
    factual_list = []
    pearsonr_list = []
    spearmanr_list = []
    non_factual_star_idx = get_non_factual_star()
    NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT",
                "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
    pos_tag = ["NOUN", "NUM", "PROPN"]
    if isinstance(loss_data[1][0][0], list):
        loss_threshold = [[[loss_word[thres_num] for loss_word in example] for example in loss_data[1]] for thres_num in range(len(loss_data[1][0][0]))]
    else:
        loss_threshold = [loss_data[1]]
    for loss_thres_idx in range(len(loss_threshold)):
        source = []
        target = []
        source_star = []
        target_star = []
        passage_score_model = []
        passage_score_human = []
        gpt3_sentences = []
        wiki_passage = []
        gpt3_text = []
        _words = []
        _loss = []
        res = defaultdict(list)
        for i in range(len(loss_data[0])):
            total_words = []
            offset = 0
            hc_passage = 0
            num_passage = 0
            hc_passage_noun = 0
            num_passage_noun = 0
            # these gpt3_text in WikibioGPT3d dataset contains some tokens in the end that not contained in the gpt3_sentences field.
            if i in [40, 46, 50]:
                offset = 3
            if i in [64, 154]:
                offset = 1
            for j, word in enumerate(loss_data[0][i]):
                if word == '\n' and loss_data[0][i][j-1] == ':':
                    begin_index = j+1
                    break
            loss = loss_threshold[loss_thres_idx][i][begin_index:-offset] if offset else loss_threshold[loss_thres_idx][i][begin_index:]
            words = loss_data[0][i][begin_index:-offset] if offset else loss_data[0][i][begin_index:]
            nums = loss_data[2][i][begin_index:-offset] if offset else loss_data[2][i][begin_index:]
            if i == 39:
                words = words[:102] + words[105:]
                loss = loss[:102] + loss[105:]
                nums = nums[:102] + nums[105:]
            sentences = hallucination_data[i]["gpt3_sentences"]
            cur = 0
            for s_idx, s in enumerate(sentences):
                sentence_loss_noun = 0 if pooling == "mean" else []
                sentence_loss_not_noun = 0
                sentence_loss = 0
                num_noun = 0
                num_not_noun = 0
                num = 0
                tmp_word = []
                tmp_loss = []
                words_list = [t for t in nlp(s)]
                for w in words_list:
                    if w.ent_type_ in NER_type or w.pos_ in pos_tag:
                        sentence_loss_noun += loss[cur] if pooling == "mean" else [loss[cur]]
                        num_noun += nums[cur]
                        tmp_word.append(w.text)
                        tmp_loss.append(loss[cur])
                    else:
                        sentence_loss_not_noun += loss[cur]
                        num_not_noun += 1
                    sentence_loss += loss[cur]
                    num += nums[cur]
                    cur += 1
                    total_words.append(w)
                if pooling == "mean":
                    res[f"{hallucination_data[i]['annotation'][s_idx]}_noun"].append(sentence_loss_noun/num_noun if num_noun else 0)
                else:
                    sorted_loss = sorted(sentence_loss_noun, reverse=True)[:top]
                    res[f"{hallucination_data[i]['annotation'][s_idx]}_noun"].append(sum(sorted_loss)/len(sorted_loss))
                res[f"{hallucination_data[i]['annotation'][s_idx]}_not_noun"].append(sentence_loss_not_noun / num_not_noun if num_not_noun else 0)
                res[f"{hallucination_data[i]['annotation'][s_idx]}"].append(sentence_loss / num)
                hc_passage += sentence_loss
                num_passage += num
                hc_passage_noun += sentence_loss_noun
                num_passage_noun += num_noun
                source.append(res[f"{hallucination_data[i]['annotation'][s_idx]}_noun"][-1] if only_noun else res[f"{hallucination_data[i]['annotation'][s_idx]}"][-1])
                # scaling scores to 0-1
                source[-1] = 1-1/source[-1]
                target.append(0 if hallucination_data[i]['annotation'][s_idx] == "accurate" else 1 if hallucination_data[i]['annotation'][s_idx] == "minor_inaccurate" else 2)
                if i in non_factual_star_idx:
                    source_star.append(source[-1])
                    target_star.append(0 if target[-1] <= 1 else 1)
                gpt3_sentences.append(s)
                wiki_passage.append(hallucination_data[i]["wiki_bio_text"])
                gpt3_text.append(hallucination_data[i]["gpt3_text"])
                _words.append(tmp_word)
                _loss.append(tmp_loss)
            passage_score_model.append((hc_passage_noun if only_noun else hc_passage)/ (num_passage_noun if only_noun else num_passage))
            passage_score_model[-1] = 1-1/passage_score_model[-1]
            passage_score_human.append(get_human_passage_score(i))
            if len(words) != cur:
                print("error!")
        non_factual_list.append(calcu_auc_pr(source, target))
        non_factual_star_list.append(calcu_auc_pr(source_star, target_star))
        factual_list.append(calcu_auc_pr_reverse(source, target))
        pearsonr_list.append(scipy.stats.pearsonr(passage_score_model, passage_score_human)[0])
        spearmanr_list.append(scipy.stats.spearmanr(passage_score_model, passage_score_human)[0])
        print(f"non_factual: {non_factual_list[-1]*100:.2f} || non_factual*: {non_factual_star_list[-1]*100:.2f} || factual: {factual_list[-1]*100:.2f} || Pearson: {pearsonr_list[-1]*100:.2f} || Spearman: {spearmanr_list[-1]*100:.2f}")
        if not only_noun:
            print("---------------------")
            break
    return res

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

def add_type(text):
    words = [t for t in nlp(text)]
    word_type = [t.ent_type_ for t in words]
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
        text = f"{text[:start]}{left_mark}{word_type[i]}{right_mark} {text[start:]}" if len(word_type[i]) and (text[start-1] == ' ' or start == 0) else text
        offset += len(f"{left_mark}{word_type[i]}{right_mark} ") if len(word_type[i]) and (text[start-1] == ' ' or start == 0) else 0
    return text

def get_non_factual_star():
    idx = []
    for i in range(len(hallucination_data)):
        flag = False
        for label in hallucination_data[i]["annotation"]:
            if label != 'major_inaccurate':
                flag = True
                break
        if flag:
            idx.append(i)
    return idx

def get_human_passage_score(idx):
    labels = []
    for label in hallucination_data[idx]["annotation"]:
        labels.append(0 if label == "accurate" else 1)
    return sum(labels)/len(labels)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/weights/llama/hf/")
    parser.add_argument("--low_cpu_mem_usage", action="store_true")
    args = parser.parse_args()
    model_paths = [args.model_path]
    for model_path in model_paths:
        if 'opt' in model_path:
            LLM = OPTForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        elif 'gpt-j' in model_path:
            LLM = GPTJForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        elif 'gpt-neox' in model_path or 'RedPajama' in model_path:
            LLM = GPTNeoXForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        elif 'Llama-2' in model_path:
            from models.modeling_llama2 import LlamaForCausalLM
            LLM = LlamaForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/huggingface_models/llama-65b", use_fast=True)
        elif 'llama' in model_path or 'vicuna' in model_path.lower() or 'TheBloke' in model_path:
            LLM = LlamaForCausalLM
            tokenizer = AutoTokenizer.from_pretrained("/home/ubuntu/huggingface_models/llama-65b", use_fast=True)
        elif 'falcon-7b' in model_path:
            from models.modeling_RW import RWForCausalLM
            LLM = RWForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        elif 'falcon-40b' in model_path:
            from models.modeling_RW_40b import RWForCausalLM
            LLM = RWForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)

        model = LLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,cache_dir="/home/ubuntu/huggingface_models",
        )
        hallucination_data = pickle.load(open("data/wikibio_gpt3_v3.pkl", "rb"))
        prompt = []
        if "ausboss" in model_path:
            instruction = [
                f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction\nPlease complete the passage below using appropriate words that follow to the given type with < > wrapped.\n\n### Response\n{add_type(d['question'])}:\n"
                for d in hallucination_data]
        else:
            instruction = [
                f"Please complete the passage below using appropriate words that follow to the given type with < > wrapped.\n{add_type(d['question'])}:\n"
                for d in hallucination_data]


        output = [d['gpt3_text'] for d in hallucination_data]
        for i in range(len(output)):
            prompt.append(f"{instruction[i]}{add_type(output[i])}")

        if 'gpt-j' in model_path or 'gpt-neox' in model_path or 'falcon' in model_path or 'RedPajama' in model_path:
            prompt = [f"<|endoftext|>{p}" for p in prompt]

        def test(p, pooling="mean", top=5, use_idf=False):
            words_list, losses_list, nums_list = [], [], []
            for i in tqdm(range(len(instruction))):
                words, losses, nums = run_generate(p[i], use_threshold=True, rm_low_prob=True, use_idf=use_idf)
                words_list.append(words)
                losses_list.append(losses)
                nums_list.append(nums)
            print("results with token type added of", model_path.split('/')[-1])
            evaluate((words_list, losses_list, nums_list), pooling=pooling, top=top)

        test(prompt, use_idf=True)
        print("------------no idf------------")
        test(prompt, use_idf=False)

        prompt = []
        instruction = [f"{d['question']}:\n" for d in hallucination_data]
        output = [d['gpt3_text'] for d in hallucination_data]
        for i in range(len(output)):
            prompt.append(f"{instruction[i]}{output[i]}")
        if 'gpt-j' in model_path or 'gpt-neox' in model_path or 'falcon' in model_path or 'RedPajama' in model_path:
            prompt = [f"<|endoftext|>{p}" for p in prompt]
        def test(p, pooling="mean", top=5):
            words_list, losses_list, nums_list = [], [], []
            for i in tqdm(range(len(instruction))):
                words, losses, nums = run_generate(p[i], use_threshold=True, rm_low_prob=False)
                words_list.append(words)
                losses_list.append(losses)
                nums_list.append(nums)
            print(model_path.split('/')[-1])
            evaluate((words_list, losses_list, nums_list), pooling=pooling, top=top, only_noun=False)
            evaluate((words_list, losses_list, nums_list), pooling=pooling, top=top, only_noun=True)
        test(prompt)
        del model
        torch.cuda.empty_cache()