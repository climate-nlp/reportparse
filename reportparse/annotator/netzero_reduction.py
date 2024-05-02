from logging import getLogger
import argparse
from distutils.util import strtobool

from transformers import AutoTokenizer, AutoModelForSequenceClassification

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification


@BaseAnnotator.register("netzero_reduction")
class NetzeroReductionAnnotator(BaseAnnotator):

    """
    This class is an annotator of https://huggingface.co/climatebert/netzero-reduction
    According to the README of the model,
    > Based on this paper, this is the fine-tuned ClimateBERT language model with a classification head
    > for detecting sentences that are either related to emission net zero or reduction targets.
    > We use the climatebert/distilroberta-base-climate-f language model as a starting point
    > and fine-tuned it on our human-annotated dataset.

    @article{schimanski2023climatebertnetzero,
        title={ClimateBERT-NetZero: Detecting and Assessing Net Zero and Reduction Targets},
        author={Tobias Schimanski and Julia Bingler and Camilla Hyslop and Mathias Kraus and Markus Leippold},
        year={2023},
        eprint={2310.08096},
        archivePrefix={arXiv},
        primaryClass={cs.LG}
    }
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.netzero_reduction_model_name_or_path = 'climatebert/netzero-reduction'
        self.netzero_reduction_tokenizer_name_or_path = 'climatebert/distilroberta-base-climate-f'
        self.sentence_climate_tokenizer = None
        self.sentence_climate_model = None
        self.block_climate_tokenizer = None
        self.block_climate_model = None
        self.sentence_climate_detection_model_name_or_path = 'ESGBERT/EnvironmentalBERT-environmental'
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

        max_len = args.netzero_reduction_max_len if args is not None else max_len
        batch_size = args.netzero_reduction_batch_size if args is not None else batch_size
        target_layouts = args.netzero_reduction_target_layouts if args is not None else target_layouts
        level = args.netzero_reduction_level if args is not None else level
        use_deprecated = args.netzero_reduction_use_deprecated if args is not None else use_deprecated

        assert level in ['block', 'sentence']
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & {'title', 'text', 'list'}

        if use_deprecated:
            logger.warning('You are using the deprecated version. We will force parameters.')
            level = 'block'
            target_layouts = ['text', 'list']

        if self.tokenizer is None or self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.netzero_reduction_tokenizer_name_or_path,
                max_len=max_len
            )
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.netzero_reduction_model_name_or_path
            )

        # Predict netzero or reduction claims
        document = annotate_by_sequence_classification(
            annotator_name='netzero_reduction',
            document=document,
            tokenizer=self.tokenizer,
            model=self.model,
            level=level,
            target_layouts=target_layouts,
            batch_size=batch_size
        )

        """
        We use an additional classifier to filter out non-environment mentions, by following the instruction of
        the original model (https://huggingface.co/climatebert/netzero-reduction):
        > IMPORTANT REMARK: It is highly recommended to use a prior classification step before applying 
        > ClimateBERT-NetZero. Establish a climate context with climatebert/distilroberta-base-climate-detector for 
        > paragraphs or ESGBERT/EnvironmentalBERT-environmental for sentences and then label the data with 
        > ClimateBERT-NetZero.
        """

        if not use_deprecated:
            if level == 'sentence' and (self.sentence_climate_tokenizer is None or self.sentence_climate_model is None):
                self.sentence_climate_tokenizer = AutoTokenizer.from_pretrained(
                    self.sentence_climate_detection_model_name_or_path,
                    max_len=max_len
                )
                self.sentence_climate_model = AutoModelForSequenceClassification.from_pretrained(
                    self.sentence_climate_detection_model_name_or_path
                )
            if level == 'block' and (self.block_climate_tokenizer is None or self.block_climate_model is None):
                self.block_climate_tokenizer = AutoTokenizer.from_pretrained(
                    self.block_climate_detection_model_name_or_path,
                    max_len=max_len
                )
                self.block_climate_model = AutoModelForSequenceClassification.from_pretrained(
                    self.block_climate_detection_model_name_or_path
                )

            # Get climate related sentences
            document_climate_annot = annotate_by_sequence_classification(
                annotator_name='dummy',
                document=document,
                tokenizer=self.sentence_climate_tokenizer if level == 'sentence' else self.block_climate_tokenizer,
                model=self.sentence_climate_model if level == 'sentence' else self.block_climate_model,
                level=level,
                target_layouts=target_layouts,
                batch_size=batch_size
            )
            climate_unrelated_object_ids = []
            for annot_obj, annot in document_climate_annot.find_annotations_by_annotator_name('dummy'):
                if annot.value == ('none' if level == 'sentence' else 'no'):
                    climate_unrelated_object_ids.append(annot_obj.id)

            # Remove annotations that do not relate to climate change or environment
            for page in document.pages:
                for block in page.blocks:
                    if level == 'block' and block.id in climate_unrelated_object_ids:
                        block.remove_annotator(annotator_name='netzero_reduction')
                        #logger.info(f'Removed the "netzero_reduction" annotation of the following block '
                        #            f'because it is not related to environment: "{block.text}".')
                        continue
                    for sentence in block.sentences:
                        if level == 'sentence' and sentence.id in climate_unrelated_object_ids:
                            sentence.remove_annotator(annotator_name='netzero_reduction')
                            #logger.info(f'Removed the "netzero_reduction" annotation of the following sentence '
                            #            f'because it is not related to environment: "{sentence.text}".')
                            continue

        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--netzero_reduction_level',
            type=str,
            default='block'
        )
        parser.add_argument(
            '--netzero_reduction_max_len',
            type=int,
            default=256
        )
        parser.add_argument(
            '--netzero_reduction_batch_size',
            type=int,
            default=8
        )
        parser.add_argument(
            '--netzero_reduction_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list']
        )
        parser.add_argument(
            '--netzero_reduction_use_deprecated',
            type=strtobool,
            help='Use the deprecated version (not recommended)',
            default=False
        )

