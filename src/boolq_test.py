import pandas as pd
import torch
import unittest
import numpy as np

from boolq import BoolQDataset
from transformers import RobertaTokenizerFast


class TestBoolQDataset(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
        self.dataset = pd.DataFrame.from_dict(
            {
                "question": ["question 0", "question 1"],
                "passage": ["passage 0", "passage 1"],
                "idx": [0, 1],
                "label": [True, False],
            }
        )
        self.max_seq_len = 4
        self.boolq_dataset = BoolQDataset(
            self.dataset, self.tokenizer, self.max_seq_len
        )

    def test_len(self):
        ## TODO: Test that the length of self.boolq_dataset is correct.
        ## len(self.boolq_dataset) should equal len(self.dataset).
        assert len(self.boolq_dataset) == len(self.dataset)

    def test_item(self):
        ## TODO: Test that, for each element of self.boolq_dataset, 
        ## the output of __getitem__ (accessible via self.boolq_dataset[idx])
        ## has the correct keys, value dimensions, and value types.
        ## Each item should have keys ["input_ids", "attention_mask", "labels"].
        ## The input_ids and attention_mask values should both have length self.max_seq_len
        ## and type torch.long. The labels value should be a single numeric value.
        for i in range(len(self.boolq_dataset)):
            boolq_item = self.boolq_dataset.__getitem__(i)
            boolq_keys = list(boolq_item.keys())
            input_ids = boolq_item['input_ids']
            attention_mask = boolq_item['attention_mask']
            label = boolq_item['labels']
            self.assertListEqual(boolq_keys, ["input_ids", "attention_mask", "labels"])
            assert len(input_ids) == self.max_seq_len
            assert len(attention_mask) == self.max_seq_len
            self.assertEqual(input_ids.dtype, torch.long)
            self.assertEqual(attention_mask.dtype, torch.long)
            print(label)
            assert isinstance(label,int)


        pass


if __name__ == "__main__":
    unittest.main()
