import re
import ftfy
import json
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
from rich import progress
import argparse
from pathlib import Path
import torch
import multiprocessing
from multiprocessing import Pool


def get_pairs(token):
    pairs = set()  # remember that order is not preserved in a set
    # Loop through the token and create 'pairs' of adjacent strings
    # e.g. "shoulder." -> ("s", "h"), ("h", "o"), ("o", "u"), ("u", "l"), ("l", "d"), ("d", "e"), ("e", "r"), ("r", ".</w>"))
    # or e.g. "shoulder." -> ("sh", "ou"), ("ou", "ld"), ("ld", "er"), ("er", ".</w>")
    previous_string = token[0]
    for string in token[1:]:
        pairs.add((previous_string, string))
        previous_string = string
    return pairs


def standardise_text(text):
    """
    The original OpenAI implementation claims this fixes some issues with SpaCy's tokenisation of BooksCorpus
    """
    text = text.replace("—", "-")
    text = text.replace("–", "-")
    text = text.replace("―", "-")
    text = text.replace("…", "...")
    text = text.replace("´", "'")
    text = re.sub(
        """(-+|~+|!+|"+|;+|\?+|\++|,+|\)+|\(+|\\+|\/+|\*+|\[+|\]+|}+|{+|\|+|_+)""",
        r" \1 ",
        text,
    )
    text = re.sub("\s*\n\s*", " \n ", text)
    text = re.sub("[^\S\n]+", " ", text)
    return text.strip()


def pad_and_truncate(lists, max_length, pad_token_id):
    padded_and_truncated_lists = []
    for list in lists:
        while len(list) > max_length:
            padded_and_truncated_lists.append(list[:max_length])
            list = list[max_length:]
        if len(list) < max_length:
            list += [pad_token_id] * (max_length - len(list))
            padded_and_truncated_lists.append(list)
    return padded_and_truncated_lists


class TextEncoder:
    def __init__(self, vocab_path, merges_path):
        self.tokeniser = Tokenizer(English().vocab)

        # Encoder maps tokens to integers
        self.encoder = json.load(open(vocab_path))
        # Decoder is a flipped encoder
        self.decoder = {v: k for k, v in self.encoder.items()}

        # In byte-pair encoding, the most frequent pair of adjacent tokens are merged into a single token 'k' times, automatically building a vocabulary of the most common tokens
        # OpenAI provide a BPE vocabulary (called 'merges', because it a list of adjacent token pairs having been 'merged'), with 40,000 items
        merges = (
            open(merges_path).read().split("\n")[1:-1]
        )  # Remove the first line (version number) and the last line (empty) of the txt file
        merges = [
            tuple(merge.split()) for merge in merges
        ]  # The merges are stored with a space, so split them into tuples

        # Map the tuples to integers. Since the original txt file is ordered by frequency, the more frequent pairs will have lower integers
        # This creates a similar shape to the vocabulary, which is also ordered by frequency, but the integer 'mappings' are not the same
        self.bpe_ranks = dict(zip(merges, range(len(merges))))

        self.cache = {}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]

        # "</w>" is used to indicate the end of a word in OpenAI's BPE vocabulary
        # Split the token into characters, and add "</w>" to the end, e.g. "shoulder." -> ("s", "h", "o", "u", "l", "d", "e", "r", ".</w>")
        word = token_by_characters = tuple(token[:-1]) + (token[-1] + "</w>",)

        pairs = get_pairs(token_by_characters)

        if not pairs:  # If the token is a single character
            return token + "</w>"

        while True:
            # TODO: add annotations and tidy
            most_frequent_pair_in_token = min(
                pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf"))
            )
            if most_frequent_pair_in_token not in self.bpe_ranks:
                # TODO: why is this?
                break
            first, second = most_frequent_pair_in_token
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)

        word = " ".join(word)
        if word == "\n  </w>":
            word = "\n</w>"
        self.cache[token] = word
        return word

    def encode(self, text):
        # Preprocess the text
        fixed_text = ftfy.fix_text(text)
        standardised_text = standardise_text(fixed_text)
        # The SpaCy tokeniser provides a third layer of standardisation, and divides the text into tokens (~words, ~punctiation, ~numbers, etc)
        preprocessed_text = self.tokeniser(standardised_text)

        text_token_ids = []
        for token in preprocessed_text:
            # A 'token' out of SpaCy might consist of multiple token ids in the vocabulary
            bpe = self.bpe(
                token.text.lower()
            ).split()  # e.g. 'shoulder.' -> ['shoul', 'der', '.</w>']

            # If the token is not in the vocabulary, return 0 (the out-of-vocabulary token id)
            token_ids = [self.encoder.get(t, 0) for t in bpe]
            text_token_ids.extend(token_ids)
        return text_token_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--merges-path", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--context-len", type=int, required=True)
    parser.add_argument("--pad-token-id", type=int, required=True) # OpenAI's vocabulary doesn't include one, but 0 (the out-of-vocabulary token id) should work (?)
    parser.add_argument("--num-processes", type=int, default=1) # One file will be saved for each process
    args = parser.parse_args()

    encoder = TextEncoder(vocab_path=args.vocab_path, merges_path=args.merges_path)

    txt_file_paths = list(Path(args.input_dir).glob("**/*.txt"))
    txt_file_paths = [txt_file_paths[i::args.num_processes] for i in range(args.num_processes)]

    def tokenize_files(txt_file_paths):
        all_token_ids = []
        for txt_file_path in progress.track(txt_file_paths, description="Tokenising..."):
            with open(txt_file_path) as f:
                text = f.read()
                token_ids = encoder.encode(text)
                all_token_ids.append(token_ids)
        all_token_ids = pad_and_truncate(all_token_ids, max_length=args.context_len, pad_token_id=args.pad_token_id)
        torch.save(torch.tensor(all_token_ids, dtype=torch.int), args.output_dir + f"bookscorpus-{multiprocessing.current_process()._identity[0] - 1}.pt")
        return all_token_ids

    with Pool(args.num_processes) as pool:
        all_token_ids = pool.map(tokenize_files, txt_file_paths)