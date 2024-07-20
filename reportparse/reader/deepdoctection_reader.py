from logging import getLogger
import argparse
import os
from typing import List

import deepdoctection.utils
import math
import numpy as np
import deepdoctection as dd
import spacy

from reportparse.reader.base import BaseReader
from reportparse.structure.document import Document, Page, Block, Table, Figure


@BaseReader.register("deepdoctection")
class DeepdoctectionReader(BaseReader):

    """
    The class for PDF file reading by deepdoctection
    """

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.en_core_web_sm = None
        return

    def _make_block(self, page: dd.Page, layout: dd.Layout) -> Block:
        if self.en_core_web_sm is None:
            self.en_core_web_sm = spacy.load('en_core_web_sm')

        text_annotation_ids = layout.text_['annotation_ids']
        text_bboxes = [page.get_annotation(annotation_ids=aid)[0].bbox for aid in text_annotation_ids]

        block_text = ''
        text_spans = []
        for text in layout.text_['text_list']:
            text_spans.append((len(block_text), len(block_text + text)))
            block_text += text + ' '

        assert len(text_spans) == len(text_annotation_ids) == len(text_bboxes)

        block = Block(
            block_id=layout.annotation_id,
            text=block_text.rstrip(),
            layout_type=layout.category_name.value,
            bbox=tuple(layout.bbox)
        )

        for text_annotation_id, text_span, text_bbox in zip(text_annotation_ids, text_spans, text_bboxes):
            block.add_text(
                span_id=text_annotation_id,
                span=text_span,
                bbox=text_bbox,
            )

        doc = self.en_core_web_sm(block_text)

        for sent in doc.sents:

            sent_bbox = [9999999, 9999999, 0, 0]
            start_text_annotation_id, end_text_annotation_id = None, None
            for text_span, text_bbox, text_annotation_id in zip(text_spans, text_bboxes, text_annotation_ids):
                if set(range(*text_span)) & set(range(sent.start_char, sent.end_char)):
                    sent_bbox[0] = min(sent_bbox[0], text_bbox[0])
                    sent_bbox[1] = min(sent_bbox[1], text_bbox[1])
                    sent_bbox[2] = max(sent_bbox[2], text_bbox[2])
                    sent_bbox[3] = max(sent_bbox[3], text_bbox[3])

                    if start_text_annotation_id is None:
                        start_text_annotation_id = text_annotation_id
                    end_text_annotation_id = text_annotation_id

            assert start_text_annotation_id is not None
            assert end_text_annotation_id is not None
            #assert sent.text.startswith(layout.text_['text_list'][text_annotation_ids.index(start_text_annotation_id)])
            #assert sent.text.endswith(layout.text_['text_list'][text_annotation_ids.index(end_text_annotation_id)])

            block.add_sentence(
                span_id=layout.annotation_id + '_sent_' + str(len(block.sentences)),
                span=(sent.start_char, sent.end_char),
                bbox=sent_bbox,
                reference={
                    'start_text_annotation_id': start_text_annotation_id,
                    'end_text_annotation_id': end_text_annotation_id,
                }
            )

        return block

    def _make_table(self, page: dd.Page, table: dd.Table) -> Table:
        table_obj = Table(table_id=table.annotation_id, html=table.html, text=table.text, bbox=tuple(table.bbox))
        for cell in table.cells:
            text_annotation_ids = cell.text_['annotation_ids']
            text_bboxes = [page.get_annotation(annotation_ids=aid)[0].bbox for aid in text_annotation_ids]

            block_text = ''
            text_spans = []
            for text in cell.text_['text_list']:
                text_spans.append((len(block_text), len(block_text + text)))
                block_text += text + ' '

            assert len(text_spans) == len(text_annotation_ids) == len(text_bboxes)

            block = Block(
                block_id=cell.annotation_id,
                text=block_text.rstrip(),
                layout_type=cell.category_name.value,
                bbox=tuple(cell.bbox)
            )

            for text_annotation_id, text_span, text_bbox in zip(text_annotation_ids, text_spans, text_bboxes):
                block.add_text(
                    span_id=text_annotation_id,
                    span=text_span,
                    bbox=text_bbox,
                )

            doc = self.en_core_web_sm(block_text)

            for sent in doc.sents:

                sent_bbox = [9999999, 9999999, 0, 0]
                start_text_annotation_id, end_text_annotation_id = None, None
                for text_span, text_bbox, text_annotation_id in zip(text_spans, text_bboxes, text_annotation_ids):
                    if set(range(*text_span)) & set(range(sent.start_char, sent.end_char)):
                        sent_bbox[0] = min(sent_bbox[0], text_bbox[0])
                        sent_bbox[1] = min(sent_bbox[1], text_bbox[1])
                        sent_bbox[2] = max(sent_bbox[2], text_bbox[2])
                        sent_bbox[3] = max(sent_bbox[3], text_bbox[3])

                        if start_text_annotation_id is None:
                            start_text_annotation_id = text_annotation_id
                        end_text_annotation_id = text_annotation_id

                assert start_text_annotation_id is not None
                assert end_text_annotation_id is not None
                #assert sent.text.startswith(layout.text_['text_list'][text_annotation_ids.index(start_text_annotation_id)])
                #assert sent.text.endswith(layout.text_['text_list'][text_annotation_ids.index(end_text_annotation_id)])

                block.add_sentence(
                    span_id=cell.annotation_id + '_sent_' + str(len(block.sentences)),
                    span=(sent.start_char, sent.end_char),
                    bbox=sent_bbox,
                    reference={
                        'start_text_annotation_id': start_text_annotation_id,
                        'end_text_annotation_id': end_text_annotation_id,
                    }
                )

            table_obj.add_block(block=block)

        return table_obj

    def analyze_image(
            self,
            pdf_path: str,
            max_pages: int = None,
            skip_pages: List[int] = None,
            skip_load_image: bool = False
    ) -> Document:

        if self.analyzer is None:
            self.analyzer = dd.get_dd_analyzer()

        df = self.analyzer.analyze(path=pdf_path, max_datapoints=max_pages)
        df.reset_state()

        document = Document(name=os.path.basename(pdf_path))

        for page in df:
            if skip_pages is not None and page.page_number in skip_pages:
                continue

            image = None if skip_load_image else page.image

            doc_page = Page(
                page_num=page.page_number, width=page.width, height=page.height, image=image
            )

            all_chunks = []
            for chunk in page._order("layouts"):
                all_chunks.append(
                    (
                        chunk.annotation_id,
                        chunk.text,
                    )
                )

            for annotation_id, text in all_chunks:
                layout = page.get_annotation(annotation_ids=annotation_id)
                assert len(layout) == 1
                layout = layout[0]

                if layout.category_name.value in ['text', 'list', 'title']:
                    if not text.strip():
                        continue
                    block = self._make_block(page=page, layout=layout)
                    doc_page.add_block(block=block)
                else:
                    raise ValueError(f'Non-supported layout type {layout.category_name.value}')

            # Add figures
            figure_annotations = [a for a in page.annotations if a.category_name.name == 'figure']
            for figure_annotation in figure_annotations:
                doc_page.add_figure(
                    figure=Figure(
                        figure_id=figure_annotation.annotation_id,
                        text=figure_annotation.text,
                        bbox=tuple(figure_annotation.bbox)),
                )

            # Add tables
            for table in page.tables:
                doc_page.add_table(table=self._make_table(page=page, table=table))

            document.add_page(page=doc_page)

            #import matplotlib.pyplot as plt
            #image = page.viz()
            #plt.figure(figsize=(25, 17))
            #plt.axis('off')
            #plt.imshow(image)

        return document

    def read(
            self,
            input_path: str, args=None,
            max_pages: int = None,
            skip_pages: List[int] = None,
            skip_load_image: bool = False
    ) -> Document:
        logger = getLogger(__name__)

        if args is None:
            logger.warning('The "read" method received the "args" argument, '
                           'which means any other optional arguments will be ignored.')

        max_pages = args.max_pages if args is not None else max_pages
        skip_pages = args.skip_pages if args is not None else skip_pages
        skip_load_image = args.skip_load_image if args is not None else skip_load_image

        document = self.analyze_image(
            pdf_path=input_path,
            max_pages=max_pages,
            skip_pages=skip_pages,
            skip_load_image=skip_load_image,
        )
        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        return

