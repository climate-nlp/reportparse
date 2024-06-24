import hashlib
import json

import pandas as pd
import numpy as np
from typing import List, Tuple, Any


def find_element_by_id(elements: list, str_id: str) -> Any:
    find = [e for e in elements if e.id == str_id]
    if find:
        if len(find) != 1:
            raise ValueError(f'Found element duplications: {find}')
        return find[0]
    return None


class Annotation:

    __slots__ = ("_annotator", "_value", "_meta", "_parent_object")

    def __init__(self, parent_object, annotator: str, value: object, meta: dict = None):
        self._parent_object = parent_object
        self._annotator = annotator
        self._value = value
        self._meta = meta

    @property
    def id(self) -> str:
        hasher = hashlib.md5()
        hasher.update(
            (self._parent_object.id + json.dumps(self.to_dict(), sort_keys=True)).encode()
        )
        return hasher.hexdigest()

    @property
    def parent_object(self) -> str:
        return self._parent_object

    def set_parent_object(self, parent_object):
        self._parent_object = parent_object

    @property
    def annotator(self) -> str:
        return self._annotator

    def set_annotator(self, annotator):
        self._annotator = annotator

    @property
    def value(self) -> object:
        return self._value

    def set_value(self, value):
        self._value = value

    @property
    def meta(self) -> dict:
        return self._meta

    def set_meta(self, meta):
        self._meta = meta

    def to_dict(self) -> dict:
        return {
            'annotator': self.annotator,
            'value': self.value,
            'meta': self.meta,
        }

    @classmethod
    def from_dict(cls, parent_object, data: dict):
        return cls(parent_object=parent_object, annotator=data['annotator'], value=data['value'], meta=data['meta'])


class Span:

    __slots__ = ("_id", "_bbox", "_span", "_parent_block", "_reference", "_annotations")

    def __init__(self, span_id: str, parent_block, span: Tuple[int, int],
                 bbox: Tuple[float, float, float, float] = None, reference: dict = None):
        self._id = span_id
        self._bbox = bbox
        self._span = span
        self._parent_block = parent_block
        self._reference = reference
        self._annotations = []

    def __str__(self):
        return self.text

    @property
    def id(self) -> str:
        return self._id

    @property
    def bbox(self) -> Tuple[float, float, float, float] or None:
        return self._bbox

    @property
    def text(self) -> str:
        return self._parent_block.text[self.span[0]: self.span[1]]

    @property
    def span(self) -> Tuple[int, int]:
        return self._span

    @property
    def reference(self) -> dict:
        return self._reference

    @property
    def annotations(self) -> List[Annotation]:
        return self._annotations

    def add_annotation(self, annotation: Annotation):
        self._annotations.append(annotation)

    def remove_annotator(self, annotator_name: str):
        self._annotations = [a for a in self._annotations if a.annotator != annotator_name]

    def remove_annotator_with_prefix(self, annotator_prefix: str):
        self._annotations = [a for a in self._annotations if not a.annotator.startswith(annotator_prefix)]

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'bbox': self.bbox,
            'span': self.span,
            'reference': self.reference,
            'annotations': [a.to_dict() for a in self.annotations]
        }

    @classmethod
    def from_dict(cls, data: dict, parent_block):
        span = cls(
            span_id=data['id'],
            parent_block=parent_block,
            span=data['span'],
            bbox=data['bbox'],
            reference=data['reference']
        )
        for annot_data in data['annotations']:
            annot = Annotation.from_dict(parent_object=span, data=annot_data)
            span.add_annotation(annot)

        return span


class Block:

    __slots__ = ("_id", "_bbox", "_text", "_layout_type", "_sentences", "_texts", "_annotations")

    def __init__(self, block_id: str, text: str,
                 layout_type: str = None, bbox: Tuple[float, float, float, float] = None):
        self._id = block_id
        self._text: str = text
        self._sentences: List[Span] = []
        self._texts: List[Span] = []
        self._bbox: Tuple[float, float, float, float] = bbox
        self._layout_type: str = layout_type
        self._annotations: List[Annotation] = []

    def __str__(self):
        return f'[{self.layout_type}] {self.text}'

    @property
    def id(self) -> str:
        return self._id

    @property
    def text(self) -> str:
        return self._text

    @property
    def bbox(self) -> Tuple[float, float, float, float] or None:
        return self._bbox

    @property
    def layout_type(self) -> str or None:
        return self._layout_type

    @property
    def sentences(self) -> List[Span]:
        return self._sentences

    def add_sentence(self, span_id: str, span: Tuple[int, int],
                     bbox: Tuple[float, float, float, float] = None, reference: dict = None):
        self._sentences.append(
            Span(span_id=span_id, parent_block=self, span=span, bbox=bbox, reference=reference)
        )

    def add_sentence_span(self, span: Span):
        self._sentences.append(span)

    @property
    def texts(self) -> List[Span]:
        return self._texts

    def add_text(self, span_id: str, span: Tuple[int, int],
                 bbox: Tuple[int, int, int, int] = None, reference: dict = None):
        self._texts.append(
            Span(span_id=span_id, parent_block=self, span=span, bbox=bbox, reference=reference)
        )

    def add_text_span(self, span: Span):
        self._texts.append(span)

    @property
    def annotations(self) -> List[Annotation]:
        return self._annotations

    def add_annotation(self, annotation: Annotation):
        self._annotations.append(annotation)

    def remove_annotator(self, annotator_name: str):
        self._annotations = [a for a in self._annotations if a.annotator != annotator_name]

    def remove_annotator_with_prefix(self, annotator_prefix: str):
        self._annotations = [a for a in self._annotations if not a.annotator.startswith(annotator_prefix)]

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'bbox': self.bbox,
            'text': self.text,
            'layout_type': self.layout_type,
            'sentences': [s.to_dict() for s in self.sentences],
            'texts': [s.to_dict() for s in self.texts],
            'annotations': [a.to_dict() for a in self.annotations],
        }

    @classmethod
    def from_dict(cls, data: dict):
        block = cls(block_id=data['id'], text=data['text'], layout_type=data['layout_type'], bbox=data['bbox'])
        for annot_data in data['annotations']:
            annot = Annotation.from_dict(parent_object=block, data=annot_data)
            block.add_annotation(annot)

        for text_data in data['texts']:
            block.add_text_span(
                span=Span.from_dict(data=text_data, parent_block=block)
            )
        for sentence_data in data['sentences']:
            block.add_sentence_span(
                span=Span.from_dict(data=sentence_data, parent_block=block)
            )
        return block

    def find_sentence_by_id(self, span_id: str) -> Span or None:
        return find_element_by_id(elements=self.sentences, str_id=span_id)

    def find_text_by_id(self, span_id: str) -> Span or None:
        return find_element_by_id(elements=self.texts, str_id=span_id)

    def find_all_annotations(self) -> List[Tuple[Span, Annotation]]:
        annots = []
        for annot in self.annotations:
            annots.append((self, annot))

        for sentence in self.sentences:
            for annot in sentence.annotations:
                annots.append((sentence, annot))

        for text in self.texts:
            for annot in text.annotations:
                annots.append((text, annot))

        return annots

    def find_annotator_names(self) -> List[str]:
        annots = self.find_all_annotations()
        annotators = sorted(set([a.annotator for o, a in annots]))
        return annotators

    def find_annotations_by_annotator_name(self, annotator_name: str) -> List[Tuple[Span, Annotation]]:
        annots = self.find_all_annotations()
        annotators = [(o, a) for o, a in annots if a.annotator == annotator_name]
        return annotators

    def find_annotation_by_id(self, annotation_id: str) -> Annotation or None:
        annots = [a for o, a in self.find_all_annotations()]
        return find_element_by_id(elements=annots, str_id=annotation_id)


class Page:

    __slots__ = ("_num", "_width", "_height", "_image", "_blocks", "_tables", "_annotations")

    def __init__(self, page_num: int, width: int, height: int, image: np.ndarray = None):
        self._num: int = page_num  # Starts with zero
        self._width: int = width
        self._height: int = height
        self._image: np.ndarray = image
        self._blocks: List[Block] = []
        self._tables: List[str] = []
        self._annotations: List[Annotation] = []

    def __str__(self):
        return f'[P.{self.num}] {self.width} x {self.height}, {len(self.blocks)} blocks and {len(self.tables)} tables'

    @property
    def id(self) -> str:
        return 'page_idx_' + str(self.num)

    @property
    def text(self) -> str:
        return '\n\n'.join([b.text for b in self.blocks])

    @property
    def num(self) -> int:
        return self._num

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def image(self) -> np.ndarray or None:
        return self._image

    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    def add_block(self, block: Block):
        self._blocks.append(block)

    @property
    def tables(self) -> List[str]:
        return self._tables

    def add_table(self, table_html):
        self._tables.append(table_html)

    @property
    def annotations(self) -> List[Annotation]:
        return self._annotations

    def add_annotation(self, annotation: Annotation):
        self._annotations.append(annotation)

    def remove_annotator(self, annotator_name: str):
        self._annotations = [a for a in self._annotations if a.annotator != annotator_name]

    def remove_annotator_with_prefix(self, annotator_prefix: str):
        self._annotations = [a for a in self._annotations if not a.annotator.startswith(annotator_prefix)]

    def to_dict(self) -> dict:
        return {
            'num': self.num,
            'width': self.width,
            'height': self.height,
            'blocks': [b.to_dict() for b in self.blocks],
            'tables': self.tables,
            'annotations': [a.to_dict() for a in self.annotations],
            #'image': self.image.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict):
        page = cls(page_num=data['num'], width=data['width'], height=data['height'], image=None)
        for annot_data in data['annotations']:
            annot = Annotation.from_dict(parent_object=page, data=annot_data)
            page.add_annotation(annot)

        for block_data in data['blocks']:
            page.add_block(
                block=Block.from_dict(data=block_data)
            )
        return page

    def find_block_by_id(self, block_id: str) -> Block:
        return find_element_by_id(elements=self.blocks, str_id=block_id)

    def find_sentence_by_id(self, span_id: str) -> Span:
        return find_element_by_id(elements=[s for b in self.blocks for s in b.sentences], str_id=span_id)

    def find_text_by_id(self, span_id: str) -> Span:
        return find_element_by_id(elements=[t for b in self.blocks for t in b.texts], str_id=span_id)

    def find_all_annotations(self) -> List[Tuple[Block or Span, Annotation]]:
        annots = []
        for annot in self.annotations:
            annots.append((self, annot))

        for block in self.blocks:
            for annot in block.annotations:
                annots.append((block, annot))

            for sentence in block.sentences:
                for annot in sentence.annotations:
                    annots.append((sentence, annot))

            for text in block.texts:
                for annot in text.annotations:
                    annots.append((text, annot))

        return annots

    def find_annotator_names(self) -> List[str]:
        annots = self.find_all_annotations()
        annotators = sorted(set([a.annotator for o, a in annots]))
        return annotators

    def find_annotations_by_annotator_name(self, annotator_name: str) -> List[Tuple[Block or Span, Annotation]]:
        annots = self.find_all_annotations()
        annotators = [(o, a) for o, a in annots if a.annotator == annotator_name]
        return annotators

    def find_annotation_by_id(self, annotation_id: str) -> Annotation or None:
        annots = [a for o, a in self.find_all_annotations()]
        return find_element_by_id(elements=annots, str_id=annotation_id)


class Document:

    __slots__ = ("_name", "_pages")

    def __init__(self, name: str):
        self._name: str = name
        self._pages: List[Page] = []

    def __str__(self):
        return f'[{self.name}] {len(self.pages)} pages'

    @property
    def name(self) -> str:
        return self._name

    @property
    def pages(self) -> List[Page]:
        return self._pages

    def find_all_annotations(self) -> List[Tuple[Page or Block or Span, Annotation]]:
        annots = []
        for page in self.pages:
            for annot in page.annotations:
                annots.append((page, annot))

            for block in page.blocks:
                for annot in block.annotations:
                    annots.append((block, annot))

                for sentence in block.sentences:
                    for annot in sentence.annotations:
                        annots.append((sentence, annot))

                for text in block.texts:
                    for annot in text.annotations:
                        annots.append((text, annot))

        return annots

    def find_annotator_names(self) -> List[str]:
        annots = self.find_all_annotations()
        annotators = sorted(set([a.annotator for o, a in annots]))
        return annotators

    def find_annotations_by_annotator_name(self, annotator_name: str) -> List[Tuple[Page or Block or Span, Annotation]]:
        annots = self.find_all_annotations()
        annotators = [(o, a) for o, a in annots if a.annotator == annotator_name]
        return annotators

    def find_annotation_by_id(self, annotation_id: str) -> Annotation or None:
        annots = [a for o, a in self.find_all_annotations()]
        return find_element_by_id(elements=annots, str_id=annotation_id)

    def add_page(self, page: Page):
        if page.num in [p.num for p in self._pages]:
            raise ValueError(f'You are trying to add existing page number.')
        self._pages.append(page)
        self._pages = sorted(self._pages, key=lambda p: p.num)

    def find_page_by_num(self, page_num: int) -> Page or None:
        find = [p for p in self._pages if p.num == page_num]
        if find:
            if len(find) != 1:
                raise ValueError(f'Found page duplications: {find}')
            return find[0]
        return None

    def find_page_by_id(self, page_id: str) -> Page or None:
        return find_element_by_id(elements=self.pages, str_id=page_id)

    def to_dict(self):
        return {
            'name': self.name,
            'pages': [p.to_dict() for p in self.pages],
        }

    def save(self, output_path: str):
        with open(output_path, 'w') as f:
            f.write(f'{json.dumps(self.to_dict(), ensure_ascii=False, indent=4)}\n')

    @classmethod
    def from_dict(cls, data: dict):
        document = cls(name=data['name'])
        for page_data in data['pages']:
            document.add_page(
                page=Page.from_dict(data=page_data)
            )
        return document

    @classmethod
    def from_json_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            jd = json.loads(f.read())
        document = Document.from_dict(data=jd)
        return document

    def to_dataframe(self, level: str) -> pd.DataFrame:
        if level not in ['page', 'block', 'sentence']:
            raise ValueError(f'The specified level ({level}) is invalid. It must be either page, block, or sentence.')

        records = []
        for page in self.pages:
            if level == 'page':
                d = {
                    'page_id': page.id,
                    'page_num': page.num,
                    'page_width': page.width,
                    'page_height': page.width,
                    'page_text': page.text,
                }
                for annot in page.annotations:
                    d[annot.annotator] = annot.value
                    if 'score' in annot.meta:
                        d[annot.annotator + '-score'] = annot.meta['score']
                records.append(d)
            elif level == 'block':
                for block in page.blocks:
                    d = {
                        'page_id': page.id,
                        'block_id': block.id,
                        'block_layout_type': block.layout_type,
                        'block_bbox': block.bbox,
                        'block_text': block.text,
                    }
                    for annot in block.annotations:
                        d[annot.annotator] = annot.value
                        if 'score' in annot.meta:
                            d[annot.annotator + '-score'] = annot.meta['score']
                    records.append(d)
            elif level == 'sentence':
                for block in page.blocks:
                    for sentence in block.sentences:
                        d = {
                            'page_id': page.id,
                            'block_id': block.id,
                            'sentence_id': sentence.id,
                            'sentence_bbox': sentence.bbox,
                            'sentence_span': sentence.span,
                            'sentence_text': sentence.text,
                        }
                        for annot in sentence.annotations:
                            d[annot.annotator] = annot.value
                            if 'score' in annot.meta:
                                d[annot.annotator + '-score'] = annot.meta['score']
                        records.append(d)

        df = pd.DataFrame(records)
        return df
