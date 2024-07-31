from typing import List, Dict, Optional, Tuple
import copy
import math
import numpy as np
import cv2
import deepdoctection as dd
from PIL import ImageColor
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class HFModelCache:
    """
    This is a singleton class to handle caching of huggingface tokenizers and models
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_tokenizer_cache'):
            self._tokenizer_cache = dict()
        if not hasattr(self, '_model_cache'):
            self._model_cache = dict()

    @property
    def current_model_cache(self) -> Dict[str, AutoModelForSequenceClassification]:
        return self._model_cache

    @property
    def current_tokenizer_cache(self) -> Dict[str, AutoTokenizer]:
        return self._tokenizer_cache

    def clear(self):
        self._model_cache = dict()
        self._tokenizer_cache = dict()
        return

    def load_tokenizer(self, tokenizer_name: str, max_len: int = 512):
        if tokenizer_name in self._tokenizer_cache:
            tokenizer = self._tokenizer_cache[tokenizer_name]
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self._tokenizer_cache[tokenizer_name] = tokenizer
        tokenizer.model_max_length = max_len
        return tokenizer

    def load_sequence_classification_model(self, model_name: str):
        if model_name in self._model_cache:
            model = self._model_cache[model_name]
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self._model_cache[model_name] = model
        return model


def draw_boxes(
    np_image,
    boxes: np.ndarray,
    category_names_list: List[Optional[str]],
    category_to_color: Optional[Dict[str, Tuple[int, int, int]]] = None,
    font_scale: float = 1.0,
    rectangle_thickness: int = 4,
):
    """
    This method was originally implemented in deepdoctection (Apache 2.0).
    https://github.com/deepdoctection/deepdoctection/blob/619f7191fa51c3886e6e5c5bda8c53c9e0e07c8d/deepdoctection/utils/viz.py#L200
    We slightly modified the origin code.
    ----
    Draw bounding boxes with category names into image.

    :param np_image: Image as np.ndarray
    :param boxes: A numpy array of shape Nx4 where each row is [x1, y1, x2, y2].
    :param category_names_list: List of N category names.
    :param category_to_color
    :param font_scale: Font scale of text box
    :param rectangle_thickness: Thickness of bounding box
    :return: A new image np.ndarray
    """
    boxes = np.asarray(boxes, dtype="int32")
    if category_names_list is not None:
        assert len(category_names_list) == len(boxes), f"{len(category_names_list)} != {len(boxes)}"
    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    sorted_inds = np.argsort(-areas)  # draw large ones first
    assert areas.min() > 0, areas.min()
    # allow equal, because we are not very strict about rounding error here
    assert (
        boxes[:, 0].min() >= 0
        and boxes[:, 1].min() >= 0
        and boxes[:, 2].max() <= np_image.shape[1]
        and boxes[:, 3].max() <= np_image.shape[0]
    ), f"Image shape: {str(np_image.shape)}\n Boxes:\n{str(boxes)}"

    np_image = np_image.copy()

    if np_image.ndim == 2 or (np_image.ndim == 3 and np_image.shape[2] == 1):
        np_image = cv2.cvtColor(np_image, cv2.COLOR_GRAY2BGR)

    for i in sorted_inds:
        box = boxes[i, :]
        if category_names_list is not None:
            choose_color = category_to_color.get(category_names_list[i])
            if font_scale > 0 and category_names_list[i] is not None:
                np_image = dd.draw_text(
                    np_image, (box[0], box[1]), category_names_list[i], color=choose_color, font_scale=font_scale
                )
            cv2.rectangle(
                np_image, (box[0], box[1]), (box[2], box[3]), color=choose_color, thickness=rectangle_thickness
            )

    return np_image


def draw_layout_on_page(
    page,
    show_annotation: bool = True,
    annotator_name: str = None,
    annotation_color: str = '#d21cff',
    show_block: bool = True,
    block_color: str = '#ff73a1',
    show_table_block: bool = True,
    table_block_color: str = '#2f73a1',
    show_figure_block: bool = True,
    figure_block_color: str = "#ff7321",
    show_sentence: bool = False,
    sentence_color: str = "#2626ff",
    show_word: bool = False,
    word_color: str = "#8a8a8a",
) -> np.ndarray:
    if page.image is not None:
        img = copy.deepcopy(page.image)
    else:
        img = np.full(
            shape=(math.ceil(page.height), math.ceil(page.width), 3),
            fill_value=255
        ).astype(np.uint8)

    if show_block:
        box_stack = []
        category_names_list = []
        for block in page.blocks:
            box_stack.append(block.bbox)
            category_names_list.append(block.layout_type)

        category_to_color = {k: ImageColor.getcolor(block_color, "RGB") for k in set(category_names_list)}

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color=category_to_color,
                font_scale=2,
                rectangle_thickness=4,
            )

    if show_table_block:
        box_stack = []
        category_names_list = []
        for block in page.table_blocks:
            box_stack.append(block.bbox)
            category_names_list.append(block.layout_type)
        category_to_color = {k: ImageColor.getcolor(table_block_color, "RGB") for k in set(category_names_list)}

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color=category_to_color,
                font_scale=2,
                rectangle_thickness=4,
            )

    if show_figure_block:
        box_stack = []
        category_names_list = []
        for figure in page.figures:
            box_stack.append(figure.bbox)
            category_names_list.append('figure')
        category_to_color = {k: ImageColor.getcolor(figure_block_color, "RGB") for k in set(category_names_list)}

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color=category_to_color,
                font_scale=2,
                rectangle_thickness=4,
            )

    if show_sentence:
        box_stack = []
        category_names_list = []
        for block in page.blocks + page.table_blocks:
            for sentence in block.sentences:
                box_stack.append(sentence.bbox)
                category_names_list.append('sentence')

        category_to_color = {'sentence': ImageColor.getcolor(sentence_color, "RGB")}

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color=category_to_color,
                font_scale=0,
                rectangle_thickness=2,
            )

    if show_word:
        box_stack = []
        category_names_list = []
        for block in page.blocks + page.table_blocks:
            for txt in block.texts:
                box_stack.append(txt.bbox)
                category_names_list.append(txt)

        category_to_color = {'word': ImageColor.getcolor(word_color, "RGB")}

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color=category_to_color,
                font_scale=0,
                rectangle_thickness=2,
            )

    if show_annotation:
        box_stack = []
        category_names_list = []
        if annotator_name is not None:
            annotations = page.find_all_annotations_by_annotator_name(annotator_name)
        else:
            annotations = page.find_all_annotations()

        for annot_obj, annot in annotations:
            if annot.annotator == 'climate_figure':
                continue
            score = f'({annot.meta["score"]:.1f})' if isinstance(annot.meta, dict) and 'score' in annot.meta else ''
            if hasattr(annot_obj, 'bbox'):
                box_stack.append(annot_obj.bbox)
                category_names_list.append(f'{annot.annotator}={annot.value}{score}')

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color={
                    k: ImageColor.getcolor(annotation_color, "RGB") for k in set(category_names_list)},
                font_scale=1.5,
                rectangle_thickness=6,
            )

    return img
