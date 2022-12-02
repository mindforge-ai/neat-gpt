import numpy as np
from torch.utils.data import Dataset


class BooksCorpus(Dataset):
    def __init__(
        self,
        pile_dir="./cache/pile",
        batch_size=32,
        min_seq_length=16,
        max_seq_length=256,
    ):
        self.pile_dir = pile_dir
        self.batch_size = batch_size
        self.min_seq_length = min_seq_length
        self.max_seq_length = max_seq_length
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        files = ["%s.jsonl" % format(_num, "02d") for _num in range(30)]
        self.file_iterator = pd.read_json(
            "/data/pile/train/%s" % files[0], lines=True, chunksize=batch_size
        )

    def __len__(self):
        return len([])

    def __getitem__(self, idx):
        raw_text = self.file_iterator[idx]
        print(raw_text)
        numpy_array = raw_text.to_numpy()

        text_data = [tokenizer.encode(text_data_) for text_data_ in numpy_array]

        min_batch_seq_length = None
        max_batch_seq_length = None
        for item in text_data:
            if min_batch_seq_length == None or len(item) < min_batch_seq_length:
                min_batch_seq_length = len(item)
            if max_batch_seq_length == None or len(item) > max_batch_seq_length:
                max_batch_seq_length = len(item)

        seq_length = (
            np.random.randint(self.max_seq_length - self.min_seq_length)
            + self.min_seq_length
        )
        seq_length = min(min_batch_seq_length - 1, seq_length)

        starting_pos = np.random.randint(
            self.max_seq_length - self.min_seq_length, size=len(text_data)
        )

        for i, pos in enumerate(starting_pos):
            if len(text_data[i]) < pos + seq_length:
                starting_pos[i] = len(text_data[i]) - seq_length - 1

        new_batch = []
        new_targets = []
        for i, text in enumerate(text_data):
            new_batch.append(text[starting_pos[i] : starting_pos[i] + seq_length])
            new_targets.append(
                text[starting_pos[i] + 1 : starting_pos[i] + seq_length + 1]
            )

        return new_batch, new_targets


print(PileDataset.__getitem__(5, 5))
