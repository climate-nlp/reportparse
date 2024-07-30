from logging import getLogger
import argparse
from distutils.util import strtobool

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document
from reportparse.util.plm_classifier import annotate_by_sequence_classification
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.util.helper import HFModelCache


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
        self.netzero_reduction_model_name_or_path = 'climatebert/netzero-reduction'
        self.netzero_reduction_tokenizer_name_or_path = 'climatebert/distilroberta-base-climate-f'
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

        assert level in LEVEL_NAMES
        assert max_len > 0
        assert batch_size > 0
        assert set(target_layouts) & LAYOUT_NAMES

        if use_deprecated:
            logger.warning('You are using the deprecated version. We will force parameters.')
            level = 'block'
            target_layouts = ['text', 'list']

        tokenizer = HFModelCache().load_tokenizer(
            self.netzero_reduction_tokenizer_name_or_path, max_len=max_len)
        model = HFModelCache().load_sequence_classification_model(
            self.netzero_reduction_model_name_or_path)

        # Predict netzero or reduction claims
        document = annotate_by_sequence_classification(
            annotator_name='netzero_reduction',
            document=document,
            tokenizer=tokenizer,
            model=model,
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
            sentence_level = level in ['sentence', 'text']
            if sentence_level:
                climate_tokenizer = HFModelCache().load_tokenizer(
                    self.sentence_climate_detection_model_name_or_path, max_len=max_len)
                climate_model = HFModelCache().load_sequence_classification_model(
                    self.sentence_climate_detection_model_name_or_path)
            else:
                climate_tokenizer = HFModelCache().load_tokenizer(
                    self.block_climate_detection_model_name_or_path, max_len=max_len)
                climate_model = HFModelCache().load_sequence_classification_model(
                    self.block_climate_detection_model_name_or_path
                )

            # Get climate related sentences
            document_climate_annot = annotate_by_sequence_classification(
                annotator_name='dummy',
                document=document,
                tokenizer=climate_tokenizer,
                model=climate_model,
                level=level,
                target_layouts=target_layouts,
                batch_size=batch_size
            )
            climate_unrelated_object_ids = []
            for annot_obj, annot in document_climate_annot.find_all_annotations_by_annotator_name('dummy'):
                if annot.value == ('none' if sentence_level else 'no'):
                    climate_unrelated_object_ids.append(annot_obj.id)

            # Remove annotations that do not relate to climate change or environment
            for page in document.pages:
                if level == 'page':
                    if page.id in climate_unrelated_object_ids:
                        page.remove_annotations_by_annotator_name(annotator_name='netzero_reduction')
                for block in page.blocks + page.table_blocks:
                    if level == 'block':
                        if block.id in climate_unrelated_object_ids:
                            block.remove_annotations_by_annotator_name(annotator_name='netzero_reduction')
                    elif level == 'sentence':
                        for sentence in block.sentences:
                            if sentence.id in climate_unrelated_object_ids:
                                sentence.remove_annotations_by_annotator_name(annotator_name='netzero_reduction')
                    elif level == 'text':
                        for text in block.texts:
                            if text.id in climate_unrelated_object_ids:
                                text.remove_annotations_by_annotator_name(annotator_name='netzero_reduction')

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
            default=['text', 'list'],
            choices=LAYOUT_NAMES
        )
        parser.add_argument(
            '--netzero_reduction_use_deprecated',
            type=strtobool,
            help='Use the deprecated version (not recommended)',
            default=False
        )

