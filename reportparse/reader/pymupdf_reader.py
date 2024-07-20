from logging import getLogger
import argparse
import os
import numpy as np
import math
import fitz
from PIL import Image
import spacy
from typing import List, Dict

from reportparse.reader.base import BaseReader
from reportparse.structure.document import Document, Page, Block


def _try_get_pixmap(pdf_path, page_number):
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_number)
    zoom = 300 / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    #img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    #img = np.array(img)


def load_dummy_page_image(page) -> Dict:
    logger = getLogger(__name__)
    logger.info('\tLoad dummy images')
    zoom = 300 / 72
    mat = fitz.Matrix(zoom, zoom)
    width, height = page.mediabox.width * zoom, page.mediabox.height * zoom
    img = np.full(shape=(math.ceil(height), math.ceil(width), 3), fill_value=255).astype(np.uint8)
    return {'success': True, 'img': img, 'mat': mat, 'width': width, 'height': height}


def load_image_from_page(page) -> Dict:
    logger = getLogger(__name__)
    logger.info(f'\tLoad images from the page {page.number}')
    zoom = 300 / 72
    mat = fitz.Matrix(zoom, zoom)

    width, height = page.mediabox.width * zoom, page.mediabox.height * zoom

    pix = page.get_pixmap(matrix=mat)
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = np.array(img)
    """
    # Workaround for a bug: https://github.com/pymupdf/PyMuPDF/issues/3072
    # (Fixed at PyMuPDF==1.23.22)
    img = np.full(shape=(math.ceil(height), math.ceil(width), 3), fill_value=255).astype(np.uint8)
    proc = multiprocessing.Process(target=_try_get_pixmap, args=(pdf_path, page.number))
    timeout = 10
    start = time.time()
    proc.start()
    success = True
    while proc.is_alive():
        time.sleep(0.01)
        end = time.time()
        if end - start > timeout:
            print("\tKilled get_pixmap because it hangs")
            proc.kill()
            success = False
            break
    if success:
        proc.kill()
        pix = page.get_pixmap(matrix=mat)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = np.array(img)
        end = time.time()
        print(f'\tLoaded image ({end - start}s)')
    """

    return {'success': True, 'img': img, 'mat': mat, 'width': width, 'height': height}


@BaseReader.register("pymupdf")
class PyMuPDFReader(BaseReader):

    """
    The class for PDF file reading by Fitz of PyMuPDF
    """

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self.en_core_web_sm = None
        return

    def _make_blocks(self, page, mat: fitz.Matrix, page_width: int, page_height: int) -> List[Block]:
        logger = getLogger(__name__)

        # word: (x0, y0, x1, y1, "word", block_no, line_no, word_no)
        words = page.get_text('words')

        if self.en_core_web_sm is None:
            self.en_core_web_sm = spacy.load('en_core_web_sm')

        all_text = ''
        word_spans = []
        word_bboxes = []
        for word in words:
            word_text = word[4]
            word_spans.append((len(all_text), len(all_text) + len(word_text)))
            all_text += word[4] + ' '
            word_bboxes.append(tuple(word[:4]))

        all_text = all_text.rstrip()

        blocks = []
        logger.info('\tApply the sentence tokenization by SpaCy')
        doc = self.en_core_web_sm(all_text)
        for i_sent, sent in enumerate(doc.sents):
            sent_bbox = [9999999., 9999999., 0., 0.]
            for word_span, text_bbox in zip(word_spans, word_bboxes):
                if set(range(*word_span)) & set(range(sent.start_char, sent.end_char)):
                    sent_bbox[0] = min(sent_bbox[0], text_bbox[0])
                    sent_bbox[1] = min(sent_bbox[1], text_bbox[1])
                    sent_bbox[2] = max(sent_bbox[2], text_bbox[2])
                    sent_bbox[3] = max(sent_bbox[3], text_bbox[3])

            # Here, a block equals to a sentence
            block_id = str(hash(f'{page.number}_{i_sent}_{sent_bbox}'))
            block_text = sent.text.strip()

            # Workaround
            sent_bbox = fitz.Rect(sent_bbox[:4])
            sent_bbox = list(sent_bbox * mat)
            sent_bbox[0] = min(sent_bbox[0], page_width)
            sent_bbox[1] = min(sent_bbox[1], page_height)
            sent_bbox[2] = min(sent_bbox[2], page_width)
            sent_bbox[3] = min(sent_bbox[3], page_height)
            sent_bbox = tuple(sent_bbox)

            block = Block(
                block_id=block_id,
                text=block_text,
                layout_type='text',
                bbox=sent_bbox
            )
            block.add_sentence(
                span_id=block_id + '_sent_0',
                span=(0, len(block_text)),
                bbox=sent_bbox,
            )

            blocks.append(block)

        return blocks

    def analyze(
            self,
            pdf_path: str,
            max_pages: int = None,
            skip_pages: List[int] = None,
            skip_load_image: bool = False
    ) -> Document:
        logger = getLogger(__name__)

        doc = fitz.open(pdf_path)

        document = Document(name=os.path.basename(pdf_path))

        for page in doc:
            logger.info(f'Read page {page.number}')
            if max_pages is not None and page.number >= max_pages:
                break
            if skip_pages is not None and page.number in skip_pages:
                continue

            if skip_load_image:
                image_info = load_dummy_page_image(page=page)
                img = None
            else:
                image_info = load_image_from_page(page=page)
                img = image_info['img']

            mat = image_info['mat']
            width = image_info['width']
            height = image_info['height']

            doc_page = Page(
                page_num=page.number,
                width=width, height=height,
                image=img,
            )

            logger.info('\tRead blocks')
            blocks = self._make_blocks(page=page, mat=mat, page_width=width, page_height=height)
            for block in blocks:
                doc_page.add_block(block=block)

            document.add_page(page=doc_page)

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

        document = self.analyze(
            pdf_path=input_path,
            max_pages=max_pages,
            skip_pages=skip_pages,
            skip_load_image=skip_load_image
        )
        return document

    def add_argument(self, parser: argparse.ArgumentParser):
        return

