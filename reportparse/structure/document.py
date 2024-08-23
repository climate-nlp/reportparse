import hashlib
import json

import pandas as pd
import numpy as np
from PIL import Image
from typing import List, Tuple, Any

from reportparse.util.settings import LAYOUT_NAMES


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

    @parent_object.setter
    def parent_object(self, parent_object):
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


class AnnotatableLevel:

    def __init__(self, _id: str, text: str, bbox: Tuple[float, float, float, float]):
        self._id = _id
        self._bbox = bbox
        self._text = text
        self._annotations: List[Annotation] = []
        return

    def __str__(self):
        raise NotImplementedError

    @property
    def id(self) -> str:
        return self._id

    @property
    def bbox(self) -> Tuple[float, float, float, float]:
        return self._bbox

    @property
    def text(self) -> str:
        return self._text

    @property
    def annotations(self) -> List[Annotation]:
        return self._annotations

    def add_annotation(self, annotation: Annotation):
        self._annotations.append(annotation)

    def remove_annotations_by_annotator_name(self, annotator_name: str):
        self._annotations = [a for a in self._annotations if a.annotator != annotator_name]

    def remove_annotations_by_annotator_prefix(self, annotator_prefix: str):
        self._annotations = [a for a in self._annotations if not a.annotator.startswith(annotator_prefix)]

    def remove_annotations(self):
        self._annotations = []

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict):
        raise NotImplementedError

    def find_block_by_id(self, block_id: str) -> 'Block':
        if hasattr(self, 'blocks'):
            return find_element_by_id(elements=self.blocks, str_id=block_id)
        else:
            raise NotImplementedError('Can not find blocks because this document level does not have them.')

    def find_sentence_by_id(self, span_id: str) -> 'Span':
        if hasattr(self, 'sentences'):
            return find_element_by_id(elements=self.sentences, str_id=span_id)
        elif hasattr(self, 'blocks'):
            return find_element_by_id(elements=[s for b in self.blocks for s in b.sentences], str_id=span_id)
        else:
            raise NotImplementedError('Can not find sentences because this document level does not have them.')

    def find_text_by_id(self, span_id: str) -> 'Span':
        if hasattr(self, 'texts'):
            return find_element_by_id(elements=self.texts, str_id=span_id)
        elif hasattr(self, 'blocks'):
            return find_element_by_id(elements=[t for b in self.blocks for t in b.texts], str_id=span_id)
        else:
            raise NotImplementedError('Can not find texts because this document level does not have them.')

    def find_all_annotations(self) -> List[Tuple['AnnotatableLevel', Annotation]]:
        annots = []
        for annot in self.annotations:
            annots.append((self, annot))

        if hasattr(self, 'blocks'):
            for block in self.blocks:
                for annot in block.annotations:
                    annots.append((block, annot))

                for sentence in block.sentences:
                    for annot in sentence.annotations:
                        annots.append((sentence, annot))

                for text in block.texts:
                    for annot in text.annotations:
                        annots.append((text, annot))
        if hasattr(self, 'tables'):
            for table in self.tables:
                for annot in table.annotations:
                    annots.append((table, annot))

                for block in table.blocks:
                    for annot in block.annotations:
                        annots.append((block, annot))

                    for sentence in block.sentences:
                        for annot in sentence.annotations:
                            annots.append((sentence, annot))

                    for text in block.texts:
                        for annot in text.annotations:
                            annots.append((text, annot))
        if hasattr(self, 'figures'):
            for figure in self.figures:
                for annot in figure.annotations:
                    annots.append((figure, annot))
        if hasattr(self, 'sentences'):
            for sentence in self.sentences:
                for annot in sentence.annotations:
                    annots.append((sentence, annot))
        if hasattr(self, 'texts'):
            for text in self.texts:
                for annot in text.annotations:
                    annots.append((text, annot))

        return annots

    def find_annotator_names(self) -> List[str]:
        annots = self.find_all_annotations()
        annotators = sorted(set([a.annotator for o, a in annots]))
        return annotators

    def find_all_annotations_by_annotator_name(self, annotator_name: str) -> List[Tuple['AnnotatableLevel', Annotation]]:
        annots = self.find_all_annotations()
        annotators = [(o, a) for o, a in annots if a.annotator == annotator_name]
        return annotators

    def find_annotation_by_id(self, annotation_id: str) -> Annotation or None:
        annots = [a for o, a in self.find_all_annotations()]
        return find_element_by_id(elements=annots, str_id=annotation_id)


class Span(AnnotatableLevel):

    __slots__ = ("_id", "_bbox", "_span", "_parent_block", "_reference", "_annotations")

    def __init__(self, span_id: str, parent_block, span: Tuple[int, int],
                 bbox: Tuple[float, float, float, float] = None, reference: dict = None):
        super().__init__(_id=span_id, text='', bbox=bbox)
        self._span = span
        self._parent_block = parent_block
        self._reference = reference

    def __str__(self):
        return self.text

    @property
    def text(self) -> str:
        return self._parent_block.text[self.span[0]: self.span[1]]

    @property
    def span(self) -> Tuple[int, int]:
        return self._span

    @property
    def parent_block(self) -> 'Block':
        return self._parent_block

    @parent_block.setter
    def parent_block(self, parent_block):
        self._parent_block = parent_block

    @property
    def reference(self) -> dict:
        return self._reference

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'bbox': self.bbox,
            'span': self.span,
            'reference': self.reference,
            'annotations': [a.to_dict() for a in self.annotations]
        }

    @classmethod
    def from_dict(cls, data: dict):
        span = cls(
            span_id=data['id'],
            parent_block=None,
            span=data['span'],
            bbox=data['bbox'],
            reference=data['reference']
        )
        for annot_data in data['annotations']:
            annot = Annotation.from_dict(parent_object=span, data=annot_data)
            span.add_annotation(annot)

        return span


class Block(AnnotatableLevel):

    __slots__ = ("_id", "_bbox", "_text", "_layout_type", "_sentences", "_texts", "_annotations")

    def __init__(self, block_id: str, text: str,
                 layout_type: str = None, bbox: Tuple[float, float, float, float] = None):
        super().__init__(_id=block_id, text=text, bbox=bbox)
        self._sentences: List[Span] = []
        self._texts: List[Span] = []
        if layout_type not in LAYOUT_NAMES:
            raise ValueError(f'{layout_type} is not the available layout type. Valid types are {LAYOUT_NAMES}')
        self._layout_type: str = layout_type

    def __str__(self):
        return f'[{self.layout_type}] {self.text}'

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
            span = Span.from_dict(data=text_data)
            span.parent_block = block
            block.add_text_span(span)
        for sentence_data in data['sentences']:
            span = Span.from_dict(data=sentence_data)
            span.parent_block = block
            block.add_sentence_span(span)
        return block


class Table(AnnotatableLevel):

    __slots__ = ("_id", "_bbox", "_html", "_text", "_blocks", "_annotations")

    def __init__(self, table_id: str, html: str, text: str, bbox: Tuple[float, float, float, float] = None):
        super().__init__(_id=table_id, text=text, bbox=bbox)
        self._html: str = html
        self._blocks: List[Block] = []

    def __str__(self):
        return f'[table] {self.text}'

    @property
    def html(self) -> str:
        return self._html

    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    def add_block(self, block: Block):
        self._blocks.append(block)

    def to_dict(self) -> dict:
        return {
            'table_id': self.id,
            'bbox': self.bbox,
            'html': self.html,
            'text': self.text,
            'blocks': [b.to_dict() for b in self.blocks],
            'annotations': [a.to_dict() for a in self.annotations],
        }

    @classmethod
    def from_dict(cls, data: dict):
        page = cls(table_id=data['table_id'], html=data['html'], text=data['text'], bbox=data['bbox'])
        for annot_data in data['annotations']:
            annot = Annotation.from_dict(parent_object=page, data=annot_data)
            page.add_annotation(annot)

        for block_data in data['blocks']:
            page.add_block(
                block=Block.from_dict(data=block_data)
            )
        return page


class Figure(AnnotatableLevel):

    __slots__ = ("_id", "_bbox", "_html", "_text", "_blocks", "_annotations")

    def __init__(self, figure_id: str, text: str, bbox: Tuple[float, float, float, float] = None):
        super().__init__(_id=figure_id, text=text, bbox=bbox)

    def __str__(self):
        return f'[figure] {self.bbox}'

    def to_dict(self) -> dict:
        return {
            'figure_id': self.id,
            'bbox': self.bbox,
            'text': self.text,
            'annotations': [a.to_dict() for a in self.annotations],
        }

    @classmethod
    def from_dict(cls, data: dict):
        page = cls(figure_id=data['figure_id'], text=data['text'], bbox=data['bbox'])
        for annot_data in data['annotations']:
            annot = Annotation.from_dict(parent_object=page, data=annot_data)
            page.add_annotation(annot)
        return page


class Page(AnnotatableLevel):

    __slots__ = ("_id", "_bbox", "_text", "_num",
                 "_width", "_height", "_image", "_blocks", "_tables", "_figures", "_annotations")

    def __init__(self, page_num: int, width: int, height: int, image: np.ndarray = None):
        super().__init__(_id='page_idx_' + str(page_num), text='', bbox=(0, 0, width, height))
        self._num: int = page_num  # Starts with zero
        self._width: int = width
        self._height: int = height
        self._image: np.ndarray = image
        self._blocks: List[Block] = []
        self._tables: List[Table] = []
        self._figures: List = []

    def __str__(self):
        return f'[P.{self.num}] {self.width} x {self.height}, {len(self.blocks)} blocks and {len(self.tables)} tables'

    @property
    def text(self) -> str:
        text = '\n\n'.join([
            b.text for b in (self.blocks + self.table_blocks)
        ])
        return text

    def get_text_by_target_layouts(self, target_layouts: List[str]) -> str:
        if not target_layouts:
            raise ValueError(f'At least one target layout must be specified from {LAYOUT_NAMES}')
        text = '\n\n'.join([
            b.text for b in (self.blocks + self.table_blocks)
            if (target_layouts and b.layout_type in target_layouts)
        ])
        return text

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

    @image.setter
    def image(self, image):
        self._image = image

    @property
    def blocks(self) -> List[Block]:
        return self._blocks

    def add_block(self, block: Block):
        self._blocks.append(block)

    @property
    def table_blocks(self) -> List[Block]:
        return [b for t in self._tables for b in t.blocks]

    @property
    def tables(self) -> List[Table]:
        return self._tables

    def add_table(self, table: Table):
        self._tables.append(table)

    @property
    def figures(self) -> List[Figure]:
        return self._figures

    def add_figure(self, figure: Figure):
        self._figures.append(figure)

    def to_dict(self) -> dict:
        return {
            'num': self.num,
            'width': self.width,
            'height': self.height,
            'blocks': [b.to_dict() for b in self.blocks],
            'tables': [t.to_dict() for t in self.tables],
            'figures': [f.to_dict() for f in self.figures],
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
            page.add_block(block=Block.from_dict(data=block_data))
        if 'tables' in data:
            for table_data in data['tables']:
                page.add_table(table=Table.from_dict(data=table_data))
        if 'figures' in data:
            for figure_data in data['figures']:
                page.add_figure(figure=Figure.from_dict(data=figure_data))
        return page

    def draw_layout(self, return_pillow_image: bool = False, **kwargs) -> np.ndarray:
        from reportparse.util.helper import draw_layout_on_page
        img = draw_layout_on_page(page=self, **kwargs)
        if return_pillow_image:
            img = Image.fromarray(img, 'RGB')
        return img

    def show(self, **kwargs):
        img = self.draw_layout(**kwargs)
        img = Image.fromarray(img, 'RGB')
        img.show()


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

    def find_all_annotations(self) -> List[Tuple[AnnotatableLevel, Annotation]]:
        annots = []
        for page in self.pages:
            annots += page.find_all_annotations()
        return annots

    def find_annotator_names(self) -> List[str]:
        annots = self.find_all_annotations()
        annotators = sorted(set([a.annotator for o, a in annots]))
        return annotators

    def find_all_annotations_by_annotator_name(self, annotator_name: str) -> List[Tuple[AnnotatableLevel, Annotation]]:
        annots = self.find_all_annotations()
        annotators = [(o, a) for o, a in annots if a.annotator == annotator_name]
        return annotators

    def find_annotation_by_id(self, annotation_id: str) -> Annotation or None:
        annots = [a for o, a in self.find_all_annotations()]
        return find_element_by_id(elements=annots, str_id=annotation_id)

    def remove_all_annotations_by_annotator_name(self, annotator_name: str):
        for annot_obj, _ in self.find_all_annotations_by_annotator_name(annotator_name=annotator_name):
            annot_obj.remove_annotations_by_annotator_name(annotator_name=annotator_name)

    def remove_all_annotations(self):
        for annot_obj, _ in self.find_all_annotations():
            annot_obj.remove_annotations()

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
        if level not in ['page', 'block', 'sentence', 'table', 'figure']:
            raise ValueError(
                f'The specified level ({level}) is invalid. It must be either page, block, table, figure, or sentence.'
            )

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
                for block in page.blocks + page.table_blocks:
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
            elif level == 'table':
                for table in page.tables:
                    d = {
                        'page_id': page.id,
                        'table_id': table.id,
                        'table_bbox': table.bbox,
                        'table_html': table.html,
                        'table_text': table.text,
                        'table_block_ids': [b.id for b in table.blocks],
                    }
                    for annot in table.annotations:
                        d[annot.annotator] = annot.value
                        if 'score' in annot.meta:
                            d[annot.annotator + '-score'] = annot.meta['score']
                    records.append(d)
            elif level == 'figure':
                for figure in page.figures:
                    d = {
                        'page_id': page.id,
                        'figure_id': figure.id,
                        'figure_bbox': figure.bbox,
                        'figure_text': figure.text,
                    }
                    for annot in figure.annotations:
                        d[annot.annotator] = annot.value
                        if 'score' in annot.meta:
                            d[annot.annotator + '-score'] = annot.meta['score']
                    records.append(d)
        df = pd.DataFrame(records)
        return df
