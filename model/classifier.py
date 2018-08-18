import torch
from typing import Optional, Dict, List
from overrides import overrides

import allennlp
from allennlp.models.model import Model
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder, FeedForward
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.training.metrics import CategoricalAccuracy, F1Measure


@Model.register("caption_classifier")
class CaptionClassifier(Model):
    def __init__(self,
                 vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 class_loss_weights: List[float],
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self.text_field_embedder = text_field_embedder
        self.encoder = encoder
        self.classifier_feedforward = classifier_feedforward

        class_loss_weights = torch.Tensor(class_loss_weights)
        class_loss_weights = class_loss_weights / class_loss_weights.sum()
        self.loss = torch.nn.CrossEntropyLoss(weight=class_loss_weights)
        
        self.metric_overall_accuracy = CategoricalAccuracy()
        self.metric_class_accuracies = {c: F1Measure(positive_label=i) for i, c in enumerate(['unfunny', 'somewhat_funny', 'funny'])}

        initializer(self)

    @overrides
    def forward(self,
                tokens: Dict[str, torch.Tensor],
                rating_probs: torch.Tensor):
        embedded = self.text_field_embedder(tokens)
        mask = allennlp.nn.util.get_text_field_mask(tokens)
        encoded = self.encoder(embedded, mask)

        logits = self.classifier_feedforward(encoded)
        output_dict = {'logits': logits}

        if rating_probs is not None:
            # which classes won?
            winning_classes = rating_probs.argmax(dim=1)

            output_dict['loss'] = self.loss(logits, winning_classes)
            self.metric_overall_accuracy(logits, winning_classes.squeeze(-1))
            for metric in self.metric_class_accuracies.values():
                metric(logits, winning_classes.squeeze(-1))

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {'overall_accuracy': self.metric_overall_accuracy.get_metric(reset)}
        for c, metric in self.metric_class_accuracies.items():
            precision, recall, f1 = metric.get_metric(reset)
            metrics[f'f1_{c}'] = f1
 
        return metrics