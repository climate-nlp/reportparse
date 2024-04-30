from logging import getLogger
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification


@BaseAnnotator.register("custom_huggingface")
class CustomHuggingfaceAnnotator(BaseAnnotator):

    """
    The custom classifier for huggingface text classification models
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        return

    def annotate(
            self,
            document: Document, args=None,
            annotator_name=None, model_name_or_path=None, tokenizer_name_or_path=None,
            max_len=128, batch_size=8, level='block', target_layouts=('text', 'list')
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        annotator_name = args.custom_huggingface_annotator_name if args is not None else annotator_name
        model_name_or_path = args.custom_huggingface_model_name_or_path if args is not None else model_name_or_path
        tokenizer_name_or_path = args.custom_huggingface_tokenizer_name_or_path \
            if args is not None else tokenizer_name_or_path
        max_len = args.custom_huggingface_max_len if args is not None else max_len
        batch_size = args.custom_huggingface_batch_size if args is not None else batch_size
        level = args.custom_huggingface_level if args is not None else level
        target_layouts = args.custom_huggingface_target_layouts if args is not None else target_layouts

        assert level in ['page', 'block', 'sentence']
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & {'title', 'text', 'list'}
        assert model_name_or_path is not None

        if annotator_name is None:
            logger.warning('You do not specify the annotator name. Will use the model name instead.')
            annotator_name = model_name_or_path

        if tokenizer_name_or_path is None:
            logger.warning('You do not specify the tokenizer name. Will use the model name instead.')
            tokenizer_name_or_path = model_name_or_path

        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, max_len=max_len)
            self.model = AutoModelForSequenceClassification.from_pretrained(tokenizer_name_or_path)

        document = annotate_by_sequence_classification(
            annotator_name=annotator_name,
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
            '--custom_huggingface_annotator_name',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--custom_huggingface_model_name_or_path',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--custom_huggingface_tokenizer_name_or_path',
            type=str,
            default=None,
        )
        parser.add_argument(
            '--custom_huggingface_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--custom_huggingface_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--custom_huggingface_level',
            type=str,
            choices=['page', 'sentence', 'block'],
            default='block'
        )
        parser.add_argument(
            '--custom_huggingface_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list']
        )

