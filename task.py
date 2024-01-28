import argparse
import spacy
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from models.modeling_opt import OPTForCausalLM
from models.modeling_llama import LlamaForCausalLM
from models.modeling_gpt_neox import GPTNeoXForCausalLM
from models.modeling_gptj import GPTJForCausalLM
from models.modeling_mistral import MistralForCausalLM
class WikiBioTask:
    def __init__(self, args):
        self.args = args
        self.prompt = lambda concept: f"This is a passage from Wikipedia about {concept}:\n"
        self.text = lambda response: f"{response}"
        self.NER_type = ['PERSON', 'DATE', 'ORG', "GPE", "NORP", 'ORDINAL', 'PRODUCT', 'CARDINAL', 'LOC', "FAC", "EVENT",
                    "WORK_OF_ART", "LAW", "LANGUAGE", "TIME", "PERCENT", "MONEY", "QUANTITY"]
        self.pos_tag = ["NOUN", "NUM", "PROPN"]
        self.nlp = spacy.load('en_core_web_sm')
        self.left_mark = '<'
        self.right_mark = '>'
        if 'opt' in args.model_path:
            LLM = OPTForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'gpt-j' in args.model_path:
            LLM = GPTJForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'gpt-neox' in args.model_path or 'RedPajama' in args.model_path:
            LLM = GPTNeoXForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'Llama-2' in args.model_path:
            from models.modeling_llama2 import LlamaForCausalLM
            LLM = LlamaForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'llama' in args.model_path or 'vicuna' in args.model_path.lower() or 'TheBloke' in args.model_path:
            LLM = LlamaForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'falcon-7b' in args.model_path:
            from models.modeling_RW import RWForCausalLM
            LLM = RWForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'falcon-40b' in args.model_path:
            from models.modeling_RW_40b import RWForCausalLM
            LLM = RWForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        elif 'mistralai' in args.model_path:
            LLM = MistralForCausalLM
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
        self.tokenizer = tokenizer
        self.model = LLM.from_pretrained(
            args.model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=args.low_cpu_mem_usage,
            cache_dir="/home/ubuntu/huggingface_models",
        )
        self.model.p = args.rho
        if not self.args.use_penalty:
            self.args.gamma = 0.

    def add_type(self, text):
        words = [t for t in self.nlp(text)]
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
            text = f"{text[:start]}{self.left_mark}{word_type[i]}{self.right_mark} {text[start:]}" if len(word_type[i]) and (
                        text[start - 1] == ' ' or start == 0) else text
            offset += len(f"{self.left_mark}{word_type[i]}{self.right_mark} ") if len(word_type[i]) and (
                        text[start - 1] == ' ' or start == 0) else 0
        return text

    def run_generate(self, prompt, gamma=0., rm_low_prob=False):
        start_token_idx = 0
        with torch.no_grad():
            encodings = self.tokenizer(prompt, return_tensors="pt")
            for i in range(len(prompt)):
                if prompt[i - 1:i + 1] == ':\n':
                    start_token_idx = encodings.char_to_token(i)
                    break
            input_ids = encodings.input_ids
            input_ids = input_ids.to(0)

            outputs = self.model(input_ids=input_ids, labels=input_ids, output_attentions=True, rm_low_prob=rm_low_prob, use_idf=self.args.use_idf)
            loss = outputs["loss"][0]
            attention = outputs["attentions"][:-1, :-1]
            attention[:start_token_idx, :] = 0
            attention[:, :start_token_idx] = 0
            attention.fill_diagonal_(0)
            mask = torch.ones_like(loss)

            words = []
            losses = []
            for sentence in self.nlp(prompt).sents:
                for span_idx, span in enumerate(sentence):
                    start, end = span.idx, span.idx + len(span)
                    cur_word = prompt[start:end]
                    if encodings.char_to_token(start) <= start_token_idx or len(cur_word.replace('\n', '')) == 0:
                        continue
                    words.append(cur_word)
                    num = 0
                    l = 0
                    memo = {}
                    for i in range(start, end):
                        loss_index = encodings.char_to_token(i) - 1
                        if loss_index not in memo:
                            if (not self.args.only_keyword) or span.text not in self.NER_type and (span.ent_type_ in self.NER_type or span.pos_ in self.pos_tag):
                                mask[loss_index] = 0
                                attention[loss_index][mask.bool()] = 0
                                weight = attention[loss_index] / (torch.sum(attention[loss_index]) + 1e-6)
                                penalty = torch.sum(weight * loss).item()
                                loss[loss_index] += gamma * penalty

                            l += loss[loss_index].item() if loss_index >= 0 else 0
                            num += 1
                            memo[loss_index] = True

                    losses.append(l / num)
                    if words[-1] == self.right_mark and (words[-2] in self.NER_type + self.pos_tag) and words[-3] == self.left_mark:
                        words = words[:-3]
                        losses = losses[:-3]
            return words, losses

    def evaluate(self, concept, response, max_score=30.):
        passage = f"Please complete the passage below using appropriate words that follow to the given type with < > wrapped.\n{self.add_type(self.prompt(concept))}{self.add_type(self.text(response))}" if self.args.add_type else f"{self.prompt(concept)}{self.text(response)}"
        sentences = [s.text.replace('\n', '') for s in self.nlp(self.text(response)).sents]
        words, losses = self.run_generate(passage, gamma=self.args.gamma, rm_low_prob=self.args.add_type)
        sentence_scores = []
        passage_score = 0
        passage_token_num = 0
        cur = 0
        for s_idx, s in enumerate(sentences):
            sentence_score = 0
            sentence_token_num = 0
            words_list = [t for t in self.nlp(s)]
            for w in words_list:
                if len(words) == cur:
                    print("error")
                if not self.args.only_keyword or w.ent_type_ in self.NER_type or w.pos_ in self.pos_tag:
                    sentence_score += losses[cur]
                    sentence_token_num += 1
                cur += 1
            passage_score += sentence_score
            passage_token_num += sentence_token_num
            sentence_scores.append(sentence_score / sentence_token_num if sentence_token_num else 0)

        sentence_score = [min(s, max_score) / max_score for s in sentence_scores]
        sentence_score = [(i, s) for i, s in enumerate(sentence_score)]
        passage_score = passage_score / passage_token_num
        passage_score = min(passage_score, max_score) / max_score
        if len(words) != cur:
            print("error")

        print(sentence_score)
        print(passage_score)
        return sentence_score, passage_score

