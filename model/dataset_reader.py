import logging
import csv
import numpy as np
from typing import Dict, List, Optional
from overrides import overrides

from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, ArrayField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.common.file_utils import cached_path

logger = logging.getLogger(__name__)


@DatasetReader.register("tny_captions")
class TNYCaptionsDatasetReader(DatasetReader):
    def __init__(self,
                 tokenizer: Optional[Tokenizer] = None,
                 token_indexers: Optional[Dict[str, TokenIndexer]] = None) -> None:
        super().__init__(False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {
            "tokens": SingleIdTokenIndexer()}

    @overrides
    def _read(self, file_path):
        file_path = cached_path(file_path)
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            next(reader)
            for row in reader:
                # skip rows from contests without absolute class counts.
                if any(row.get(x) == '' for x in ['unfunny', 'somewhat_funny', 'funny']):
                    continue

                rating_counts = [int(float(row[r]))
                                 for r in ['unfunny', 'somewhat_funny', 'funny']]
                yield self.text_to_instance(row['caption'], rating_counts)

    @overrides
    def text_to_instance(self, text: str, rating_counts: List[int]) -> Instance:
        tokens = self._tokenizer.tokenize(text)
        text_field = TextField(tokens, token_indexers=self._token_indexers)

        rating_counts = np.array(rating_counts)
        rating_probs = rating_counts / rating_counts.sum()

        return Instance({'tokens': text_field,
                         'rating_probs': ArrayField(rating_probs)})
