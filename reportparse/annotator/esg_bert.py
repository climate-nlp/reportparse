from logging import getLogger
import argparse

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.util.helper import HFModelCache


@BaseAnnotator.register("esg_bert")
class ESGBertAnnotator(BaseAnnotator):

    """
    This class is an annotator of https://huggingface.co/nbroad/ESG-BERT
    According to the README of the model,
    > Domain Specific BERT Model for Text Mining in Sustainable Investing
    """

    def __init__(self):
        super().__init__()
        self.esg_bert_model_name_or_path = 'nbroad/ESG-BERT'
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

        max_len = args.esg_bert_max_len if args is not None else max_len
        batch_size = args.esg_bert_batch_size if args is not None else batch_size
        level = args.esg_bert_level if args is not None else level
        target_layouts = args.esg_bert_target_layouts if args is not None else target_layouts

        assert level in LEVEL_NAMES
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & LAYOUT_NAMES

        tokenizer = HFModelCache().load_tokenizer(
            self.esg_bert_model_name_or_path, max_len=max_len)
        model = HFModelCache().load_sequence_classification_model(
            self.esg_bert_model_name_or_path)

        document = annotate_by_sequence_classification(
            annotator_name='esg_bert',
            document=document,
            tokenizer=tokenizer,
            model=model,
            level=level,
            target_layouts=target_layouts,
            batch_size=batch_size
        )
        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--esg_bert_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--esg_bert_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--esg_bert_level',
            type=str,
            choices=['sentence', 'block'],
            default='sentence'
        )
        parser.add_argument(
            '--esg_bert_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list'],
            choices=LAYOUT_NAMES
        )
