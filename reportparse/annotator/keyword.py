from logging import getLogger
import argparse
import re

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES


def count_keyword(pattern, text):
    return len([_ for _ in re.finditer(pattern, text)])


@BaseAnnotator.register("keyword")
class KeywordAnnotator(BaseAnnotator):

    def __init__(self):
        super().__init__()
        return

    def annotate(
            self,
            document: Document, args=None,
            search_text='green', annotator_name='keyword-green', level='sentence',
            target_layouts=('text', 'list', 'cell'),
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        search_text = args.keyword_search_text if args is not None else search_text
        annotator_name = args.keyword_annotator_name if args is not None else annotator_name
        level = args.keyword_level if args is not None else level
        target_layouts = args.keyword_target_layouts if args is not None else list(target_layouts)

        assert search_text.strip()
        assert annotator_name.strip()
        assert level in LEVEL_NAMES
        assert set(target_layouts) & LAYOUT_NAMES

        def _annotate(_annotate_obj: AnnotatableLevel, _text: str):
            _score = count_keyword(pattern=search_text, text=_text)
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=annotator_name,
                    value='yes' if _score > 0 else 'no',
                    meta={'score': _score}
                )
            )

        for page in document.pages:
            if level == 'page':
                text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                _annotate(_annotate_obj=page, _text=text)
            else:
                for block in page.blocks + page.table_blocks:
                    if target_layouts is not None and block.layout_type not in target_layouts:
                        continue
                    if level == 'block':
                        _annotate(_annotate_obj=block, _text=block.text)
                    elif level == 'sentence':
                        for sentence in block.sentences:
                            _annotate(_annotate_obj=sentence, _text=sentence.text)
                    elif level == 'text':
                        for text in block.texts:
                            _annotate(_annotate_obj=text, _text=text.text)

        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--keyword_search_text',
            type=str,
            default='\bgreen\b',
            help='Regular expression of keyword matching'
        )
        parser.add_argument(
            '--keyword_annotator_name',
            type=str,
            default='keyword'
        )
        parser.add_argument(
            '--keyword_level',
            type=str,
            choices=['sentence', 'block'],
            default='sentence'
        )
        parser.add_argument(
            '--keyword_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list', 'cell'],
            choices=LAYOUT_NAMES
        )
