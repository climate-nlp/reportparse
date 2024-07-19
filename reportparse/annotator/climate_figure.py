from logging import getLogger
import argparse

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document, Annotation
from reportparse.util.plm_classifier import annotate_by_sequence_classification


@BaseAnnotator.register("climate_figure")
class ClimateFigureAnnotator(BaseAnnotator):

    """
    This class is an annotator for detecting figures that contain climate-related text.
    To get the climate-related text, we use climatebert/distilroberta-base-climate-detector.
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
        self.tokenizer = None
        self.model = None
        self.climate_model_name_or_path = 'climatebert/distilroberta-base-climate-detector'
        return

    def annotate(
            self,
            document: Document, args=None,
            max_len=128, batch_size=8, score_threshold=0.75,
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        max_len = args.climate_figure_max_len if args is not None else max_len
        batch_size = args.climate_figure_batch_size if args is not None else batch_size
        score_threshold = args.climate_figure_score_threshold if args is not None else score_threshold

        assert max_len > 0
        assert batch_size > 0
        assert 1 >= score_threshold >= 0

        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.climate_model_name_or_path,
                max_len=max_len
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.climate_model_name_or_path
            )

        '''
        Algorithm for detecting climate-related figures:
        1. Detect climate-related blocks.
        2. Assume pages are climate-related if the climate-related blocks included in the pages are the majority.
        3. Classify any figures of the climate-related pages into the 'climate-related figure'
        '''
        document = annotate_by_sequence_classification(
            annotator_name='dummy',
            document=document,
            tokenizer=self.tokenizer,
            model=self.model,
            level='block',
            target_layouts=['text', 'list'],
            batch_size=batch_size
        )
        for page in document.pages:
            n_climate_blocks = len(
                [a for o, a in page.find_all_annotations_by_annotator_name('dummy') if a.value == 'yes'])
            n_non_climate_blocks = len(
                [a for o, a in page.find_all_annotations_by_annotator_name('dummy') if a.value == 'no'])
            n_total = n_climate_blocks + n_non_climate_blocks
            score = n_climate_blocks / n_total if n_total > 0 else 0
            is_climate_figure = score >= score_threshold
            for figure in page.figures:
                annot_obj = Annotation(
                    parent_object=figure,
                    annotator='climate_figure',
                    value='yes' if is_climate_figure else 'no',
                    meta={'score': score}
                )
                figure.add_annotation(annot_obj)

        document.remove_annotations_by_annotator_name('dummy')

        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--climate_figure_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--climate_figure_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--climate_figure_score_threshold',
            type=float,
            default=0.75
        )
