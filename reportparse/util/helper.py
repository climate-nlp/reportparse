from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2
import deepdoctection as dd
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
