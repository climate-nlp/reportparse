from logging import getLogger
import argparse
import os
import spacy
from typing import List, Dict
from typing import Callable
#import sec_parser as sp

from reportparse.reader.base import BaseReader
from reportparse.structure.document import Document, Page, Block, Table

"""
The Edgar10KParser class is modified class of Edgar10QParser here:
https://github.com/alphanome-ai/sec-parser/blob/main/sec_parser/processing_engine/core.py

We modified default steps to avoid text merges.

The license information of the original code is available here:

MIT License
Copyright (c) 2023 Alphanome.AI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from sec_parser.processing_steps.empty_element_classifier import EmptyElementClassifier
from sec_parser.processing_steps.highlighted_text_classifier import (
    HighlightedTextClassifier,
)
from sec_parser.processing_steps.image_classifier import ImageClassifier
from sec_parser.processing_steps.individual_semantic_element_extractor.individual_semantic_element_extractor import (
    IndividualSemanticElementExtractor,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.image_check import (
    ImageCheck,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.table_check import (
    TableCheck,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.top_section_title_check import (
    TopSectionTitleCheck,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.xbrl_tag_check import (
    XbrlTagCheck,
)
from sec_parser.processing_steps.introductory_section_classifier import (
    IntroductorySectionElementClassifier,
)
from sec_parser.processing_steps.page_header_classifier import PageHeaderClassifier
from sec_parser.processing_steps.page_number_classifier import PageNumberClassifier
from sec_parser.processing_steps.supplementary_text_classifier import (
    SupplementaryTextClassifier,
)
from sec_parser.processing_steps.table_classifier import TableClassifier
from sec_parser.processing_steps.table_of_contents_classifier import (
    TableOfContentsClassifier,
)
from sec_parser.processing_steps.text_classifier import TextClassifier
from sec_parser.processing_steps.title_classifier import TitleClassifier
from sec_parser.processing_steps.top_section_manager_for_10q import (
    TopSectionManagerFor10Q,
)
from sec_parser.semantic_elements.highlighted_text_element import HighlightedTextElement
from sec_parser.semantic_elements.semantic_elements import (
    NotYetClassifiedElement,
    TextElement,
)
from sec_parser.semantic_elements.table_element.table_element import TableElement

from sec_parser.processing_steps.abstract_classes.abstract_processing_step import (
    AbstractProcessingStep,
)
from sec_parser.processing_steps.individual_semantic_element_extractor.single_element_checks.abstract_single_element_check import (
    AbstractSingleElementCheck,
)
from sec_parser.processing_engine.core import AbstractSemanticElementParser


class Edgar10KParser(AbstractSemanticElementParser):

    def get_default_steps(
        self,
        get_checks: Callable[[], list[AbstractSingleElementCheck]] | None = None,
    ) -> list[AbstractProcessingStep]:
        return [
            IndividualSemanticElementExtractor(
                get_checks=get_checks or self.get_default_single_element_checks,
            ),
            ImageClassifier(types_to_process={NotYetClassifiedElement}),
            EmptyElementClassifier(types_to_process={NotYetClassifiedElement}),
            TableClassifier(types_to_process={NotYetClassifiedElement}),
            TableOfContentsClassifier(types_to_process={TableElement}),
            TopSectionManagerFor10Q(types_to_process={NotYetClassifiedElement}),
            IntroductorySectionElementClassifier(),
            TextClassifier(types_to_process={NotYetClassifiedElement}),
            HighlightedTextClassifier(types_to_process={TextElement}),
            SupplementaryTextClassifier(
                types_to_process={TextElement, HighlightedTextElement},
            ),
            PageHeaderClassifier(
                types_to_process={TextElement, HighlightedTextElement},
            ),
            PageNumberClassifier(
                types_to_process={TextElement, HighlightedTextElement},
            ),
            TitleClassifier(types_to_process={HighlightedTextElement}),
        ]

    def get_default_single_element_checks(self) -> list[AbstractSingleElementCheck]:
        return [
            TableCheck(),
            XbrlTagCheck(),
            ImageCheck(),
            TopSectionTitleCheck(),
        ]


@BaseReader.register("10k")
class Sec10KReader(BaseReader):

    """
    The class for txt file reading for 10-K document
    """

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.en_core_web_sm = None
        return

    def _make_block(self, block_id: str, text: str, element_name: str) -> Block:
        if self.en_core_web_sm is None:
            self.en_core_web_sm = spacy.load('en_core_web_sm')

        element_name_2_layout_type = {
            'TopSectionTitle': 'title',
            'TitleElement': 'title',
            'TextElement': 'text',
            'TableCell': 'cell',
        }

        block = Block(
            block_id=block_id,
            text=text.strip(),
            layout_type=element_name_2_layout_type[element_name],
            bbox=(0, 0, 0, 0)  # No bbox information available
        )

        doc = self.en_core_web_sm(text)

        for sent in doc.sents:
            block.add_sentence(
                span_id=block_id + '_sent_' + str(len(block.sentences)),
                span=(sent.start_char, sent.end_char),
                bbox=(0, 0, 0, 0)  # No bbox information available
            )
        return block

    def _make_table(self, table_id: str, bs4_tag) -> Table:
        if self.en_core_web_sm is None:
            self.en_core_web_sm = spacy.load('en_core_web_sm')

        table = Table(
            table_id=table_id,
            html=str(bs4_tag),
            text=bs4_tag.text.strip(),
            bbox=(0, 0, 0, 0)  # No bbox information available
        )

        for tr in bs4_tag.findAll('tr'):
            for td in tr.findAll('td'):
                block_id = str(hash(f'{table_id}_{len(table.blocks)}_TableCell'))
                table_text = td.getText().strip()
                if table_text:
                    table.add_block(
                        self._make_block(block_id=block_id, text=table_text, element_name='TableCell')
                    )

        return table

    def analyze(
            self,
            input_path: str,
    ) -> Document:

        with open(input_path, 'r') as f:
            html = f.read()

        elements: list = Edgar10KParser().parse(html)

        document = Document(name=os.path.basename(input_path))

        # We consider the sub top-level (i.e., level 1) section as a page
        page_num = 0
        doc_page = None
        for element in elements:
            element_info = element.to_dict()
            element_name = element_info['cls_name']
            element_level = None if 'level' not in element_info else element_info['level']

            if element_name == 'TopSectionTitle':
                if element_level == 1:
                    doc_page = Page(
                        page_num=page_num,
                        width=0, height=0,
                        image=None,
                    )
                    document.add_page(page=doc_page)
                    page_num += 1

            if doc_page is not None:
                if element_name in ['TopSectionTitle', 'TitleElement', 'TextElement']:
                    block_id = str(hash(f'{page_num}_{len(doc_page.blocks)}_{element_name}'))
                    doc_page.add_block(
                        self._make_block(block_id=block_id, text=element.text, element_name=element_name)
                    )
                elif element_name == 'TableElement':
                    table_id = str(hash(f'{page_num}_{len(doc_page.tables)}_{element_name}'))
                    doc_page.add_table(
                        self._make_table(table_id=table_id, bs4_tag=element.html_tag._bs4)
                    )

        return document

    def read(
            self,
            input_path: str, args=None,
            max_pages: int = None,
            skip_pages: List[int] = None,
            skip_load_image: bool = True
    ) -> Document:
        logger = getLogger(__name__)

        logger.warning('We ignore any arguments except input_path for this 10K reader.')
        logger.warning('We do not apply layout analysis because the input data is the HTML file. '
                       'Note that bounding boxes for any elements are always (0, 0, 0, 0) '
                       'because we do not conduct the layout analysis.')

        document = self.analyze(
            input_path=input_path,
        )
        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        return

