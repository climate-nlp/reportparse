from logging import getLogger
import argparse

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.util.helper import HFModelCache


@BaseAnnotator.register("climate")
class ClimateAnnotator(BaseAnnotator):

    """
    This class is an annotator of climatebert/distilroberta-base-climate-detector
    According to the README of the model,
    > This is the fine-tuned ClimateBERT language model with a classification head
    > for detecting climate-related paragraphs.

    @techreport{bingler2023cheaptalk,
        title={How Cheap Talk in Climate Disclosures Relates to Climate Initiatives, Corporate Emissions, and Reputation Risk},
        author={Bingler, Julia and Kraus, Mathias and Leippold, Markus and Webersinke, Nicolas},
        type={Working paper},
        institution={Available at SSRN 3998435},
        year={2023}
    }
    """

    def __init__(self):
        super().__init__()
        self.climate_model_name_or_path = 'climatebert/distilroberta-base-climate-detector'
        return

    def annotate(
            self,
            document: Document, args=None,
            max_len=128, batch_size=8, level='block', target_layouts=('text', 'list'),
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        max_len = args.climate_max_len if args is not None else max_len
        batch_size = args.climate_batch_size if args is not None else batch_size
        level = args.climate_level if args is not None else level
        target_layouts = args.climate_target_layouts if args is not None else target_layouts

        assert level in LEVEL_NAMES
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & LAYOUT_NAMES

        if level != 'block':
            logger.warning('The model is trained on paragraphs (similar to blocks). '
                           'It may not perform well on sentences.')

        tokenizer = HFModelCache().load_tokenizer(self.climate_model_name_or_path, max_len=max_len)
        model = HFModelCache().load_sequence_classification_model(self.climate_model_name_or_path)

        document = annotate_by_sequence_classification(
            annotator_name='climate',
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
            '--climate_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--climate_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--climate_level',
            type=str,
            choices=['sentence', 'block'],
            default='block'
        )
        parser.add_argument(
            '--climate_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list'],
            choices=LAYOUT_NAMES
        )

