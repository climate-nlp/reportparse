from logging import getLogger
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification


@BaseAnnotator.register("environmental_claim")
class EnvironmentalClaimAnnotator(BaseAnnotator):

    """
    This class is an annotator of https://huggingface.co/climatebert/environmental-claims
    According to the README of the model,
    > The environmental-claims model is fine-tuned on the EnvironmentalClaims dataset by using the
    > climatebert/distilroberta-base-climate-f model as pre-trained language model.
    > The underlying methodology can be found in our research paper.

    @misc{stammbach2022environmentalclaims,
        title = {A Dataset for Detecting Real-World Environmental Claims},
        author = {Stammbach, Dominik and Webersinke, Nicolas and Bingler,
        Julia Anna and Kraus, Mathias and Leippold, Markus},
        year = {2022},
        doi = {10.48550/ARXIV.2209.00507},
        url = {https://arxiv.org/abs/2209.00507},
        publisher = {arXiv},
    }
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.environmental_claim_model_name_or_path = 'climatebert/environmental-claims'
        return

    def annotate(
        self,
        document: Document, args=None,
        max_len=128, batch_size=8, level='sentence', target_layouts=('text', 'list')
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        max_len = args.environmental_claim_max_len if args is not None else max_len
        batch_size = args.environmental_claim_batch_size if args is not None else batch_size
        level = args.environmental_claim_level if args is not None else level
        target_layouts = args.environmental_claim_target_layouts if args is not None else target_layouts

        assert level in ['block', 'sentence']
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & {'title', 'text', 'list'}

        if level != 'sentence':
            logger.warning('This model is trained on sentences. It may not perform well on blocks.')

        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.environmental_claim_model_name_or_path,
                max_len=max_len
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.environmental_claim_model_name_or_path
            )

        document = annotate_by_sequence_classification(
            annotator_name='environmental_claim',
            document=document,
            tokenizer=self.tokenizer,
            model=self.model,
            level=level,
            target_layouts=target_layouts,
            batch_size=batch_size
        )
        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--environmental_claim_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--environmental_claim_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--environmental_claim_level',
            type=str,
            choices=['sentence', 'block'],
            default='sentence'
        )
        parser.add_argument(
            '--environmental_claim_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list']
        )

