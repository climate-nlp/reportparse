from logging import getLogger
import argparse
from typing import List

from reportparse.annotator.base import BaseAnnotator
from reportparse.structure.document import Document, AnnotatableLevel, Annotation
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES
from reportparse.annotator.keyword import count_keyword


@BaseAnnotator.register("standard_keyword")
class StandardKeywordAnnotator(BaseAnnotator):

    def __init__(self):
        super().__init__()
        self.standard2keywords = {
            'TCFD': [r'\bTCFD\b', 'Task Force on Climate-Related Financial Disclosures'],
            'GRI': [r'\bGRI\b', 'Global Reporting Initiative'],
            'SASB': [r'\bSASB\b', 'Sustainability Accounting Standards Board'],
            'UN Global Compact': ['Global Compact'],
            'SDGs': [r'\bSDGs\b', 'Sustainable Development Goals'],
            'ISO-14001': [r'\bISO 14001\b', r'\bISO14001\b'],
            'ISO-14040': [r'\bISO 14040\b', r'\bISO114040\b'],
            'ISO-45001': [r'\bISO 45001\b', r'\bISO45001\b'],
            'EPA': [r'\bEPA\b', 'Environmental Protection Agency'],
            'CDP': [r'\bCDP\b', 'Carbon Disclosure Project'],
            'EU REACH': [r'\bEU REACH\b', 'Registration, Evaluation, Authorisation and Restriction of Chemicals',
                         r'\bREACH\b'],
            'ICH': [r'\bICH\b' 'International Council for Harmonisation of Technical Requirements'],
            'WEF': [r'\bWEF\b', 'World Economic Forum'],
            'EcoVadis': [r'\bEcoVadis\b', r'\bEco Vadis\b', r'\becovadis\b']
        }
        return

    def annotate(
            self,
            document: Document, args=None,
            level='sentence', target_layouts=('text', 'list', 'cell'),
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "annotate" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        level = args.standard_keyword_level if args is not None else level
        target_layouts = args.standard_keyword_target_layouts if args is not None else list(target_layouts)

        assert level in LEVEL_NAMES
        assert set(target_layouts) & LAYOUT_NAMES

        def _annotate(_annotate_obj: AnnotatableLevel, _standard: str, _text: str, _keywords: List[str]):
            _score = 0
            for _keyword in _keywords:
                _score += count_keyword(pattern=_keyword, text=_text)
            _annotate_obj.add_annotation(
                annotation=Annotation(
                    parent_object=_annotate_obj,
                    annotator=f'standard_keyword-{_standard}',
                    value='yes' if _score > 0 else 'no',
                    meta={'score': _score}
                )
            )

        for standard, keywords in self.standard2keywords.items():
            for page in document.pages:
                if level == 'page':
                    text = page.get_text_by_target_layouts(target_layouts=target_layouts)
                    _annotate(_annotate_obj=page, _standard=standard, _text=text, _keywords=keywords)
                else:
                    for block in page.blocks + page.table_blocks:
                        if target_layouts is not None and block.layout_type not in target_layouts:
                            continue
                        if level == 'block':
                            _annotate(_annotate_obj=block, _standard=standard, _text=block.text, _keywords=keywords)
                        elif level == 'sentence':
                            for sentence in block.sentences:
                                _annotate(_annotate_obj=sentence, _standard=standard,
                                          _text=sentence.text, _keywords=keywords)
                        elif level == 'text':
                            for text in block.texts:
                                _annotate(_annotate_obj=text, _standard=standard,
                                          _text=text.text, _keywords=keywords)

        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        parser.add_argument(
            '--standard_keyword_level',
            type=str,
            choices=['sentence', 'block'],
            default='sentence'
        )
        parser.add_argument(
            '--standard_keyword_target_layouts',
            type=str,
            nargs='+',
            default=['text', 'list', 'cell'],
            choices=LAYOUT_NAMES
        )
