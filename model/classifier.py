from typing import Optional

from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.nn.regularizers.regularizer_applicator import RegularizerApplicator


@Model.register("caption_classifier")
class AcademicPaperClassifier(Model):
    def __init__(self, vocab: Vocabulary, regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)
