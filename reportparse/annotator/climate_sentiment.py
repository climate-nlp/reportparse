from logging import getLogger
import argparse
from distutils.util import strtobool

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES


@BaseAnnotator.register("climate_sentiment")
class ClimateSentimentAnnotator(BaseAnnotator):

    """
    This class is an annotator of https://huggingface.co/climatebert/distilroberta-base-climate-sentiment
    According to the README of the model,
    > This is the fine-tuned ClimateBERT language model with a classification head for classifying climate-related
    > paragraphs into the climate-related sentiment classes opportunity, neutral, or risk.

    @techreport{bingler2023cheaptalk,
        title={How Cheap Talk in Climate Disclosures Relates to Climate Initiatives,
        Corporate Emissions, and Reputation Risk},
        author={Bingler, Julia and Kraus, Mathias and Leippold, Markus and Webersinke, Nicolas},
        type={Working paper},
        institution={Available at SSRN 3998435},
        year={2023}
    }
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.climate_sentiment_model_name_or_path = 'climatebert/distilroberta-base-climate-sentiment'
        self.block_climate_tokenizer = None
        self.block_climate_model = None
        self.block_climate_detection_model_name_or_path = 'climatebert/distilroberta-base-climate-detector'
        return

    def annotate(
            self,
            document: Document, args=None,
            max_len=128, batch_size=8, level='block', target_layouts=('text', 'list'), use_deprecated=False,
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        max_len = args.climate_sentiment_max_len if args is not None else max_len
        batch_size = args.climate_sentiment_batch_size if args is not None else batch_size
        level = args.climate_sentiment_level if args is not None else level
        target_layouts = args.climate_sentiment_target_layouts if args is not None else target_layouts
        use_deprecated = args.climate_sentiment_use_deprecated if args is not None else use_deprecated

        if use_deprecated:
            logger.warning('You are using the deprecated version. We will force parameters.')
            level = 'block'
            target_layouts = ['text', 'list']

        assert level in LEVEL_NAMES
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & LAYOUT_NAMES

        if level != 'block':
            logger.warning('The model is trained on paragraphs (similar to blocks). '
                           'It may not perform well on sentences.')

        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.climate_sentiment_model_name_or_path,
                max_len=max_len
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.climate_sentiment_model_name_or_path
            )

        document = annotate_by_sequence_classification(
            annotator_name='climate_sentiment',
            document=document,
            tokenizer=self.tokenizer,
            model=self.model,
            level=level,
            target_layouts=target_layouts,
            batch_size=batch_size
        )

        """
        We use an additional classifier to filter out non-climate mentions
        """

        if not use_deprecated:
            if self.block_climate_tokenizer is None or self.block_climate_model is None:
                self.block_climate_tokenizer = AutoTokenizer.from_pretrained(
                    self.block_climate_detection_model_name_or_path,
                    max_len=max_len
                )
                self.block_climate_model = AutoModelForSequenceClassification.from_pretrained(
                    self.block_climate_detection_model_name_or_path
                )

            # Get climate related mentions
            document_climate_annot = annotate_by_sequence_classification(
                annotator_name='dummy',
                document=document,
                tokenizer=self.block_climate_tokenizer,
                model=self.block_climate_model,
                level=level,
                target_layouts=target_layouts,
                batch_size=batch_size
            )
            climate_unrelated_object_ids = []
            for annot_obj, annot in document_climate_annot.find_all_annotations_by_annotator_name('dummy'):
                if annot.value == 'no':
                    climate_unrelated_object_ids.append(annot_obj.id)

            # Remove annotations that do not relate to climate
            for page in document.pages:
                for block in page.blocks:
                    if level == 'block' and block.id in climate_unrelated_object_ids:
                        block.remove_annotations_by_annotator_name(annotator_name='climate_sentiment')
                        #logger.info(f'Removed the "climate_sentiment" annotation of the following block '
                        #            f'because it is not related to environment: "{block.text}".')
                        continue
                    for sentence in block.sentences:
                        if level == 'sentence' and sentence.id in climate_unrelated_object_ids:
                            sentence.remove_annotations_by_annotator_name(annotator_name='climate_sentiment')
                            #logger.info(f'Removed the "climate_sentiment" annotation of the following sentence '
                            #            f'because it is not related to environment: "{sentence.text}".')
                            continue

        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--climate_sentiment_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--climate_sentiment_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--climate_sentiment_level',
            type=str,
            choices=['sentence', 'block'],
            default='block'
        )
        parser.add_argument(
            '--climate_sentiment_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list'],
            choices=LAYOUT_NAMES
        )
        parser.add_argument(
            '--climate_sentiment_use_deprecated',
            type=strtobool,
            help='Use the deprecated version (not recommended)',
            default=False
        )

