import copy
from typing import List
from tqdm.auto import tqdm

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

from reportparse.structure.document import Document, Annotation
from reportparse.util.settings import LAYOUT_NAMES, LEVEL_NAMES


def _load_dataset(document: Document, level: str, target_layouts: List[str] = None) -> datasets.Dataset:
    if level not in LEVEL_NAMES:
        raise ValueError(f'The specified level ({level}) is invalid. It must be one of {LEVEL_NAMES}.')

    data_points = []

    for page in document.pages:

        if level == 'page':
            data_points.append({
                'level': level,
                'document_id': document.name,
                'text': page.get_text_by_target_layouts(target_layouts=target_layouts) if target_layouts else page.text,
                'page_num': page.num,
            })
        else:
            for block in page.blocks + page.table_blocks:
                if target_layouts is not None and block.layout_type not in target_layouts:
                    continue
                if level == 'block':
                    data_points.append({
                        'level': level,
                        'document_id': document.name,
                        'text': block.text,
                        'page_num': page.num,
                        'id': block.id,
                    })
                elif level == 'sentence':
                    for sentence in block.sentences:
                        data_points.append({
                            'level': level,
                            'document_id': document.name,
                            'text': sentence.text,
                            'page_num': page.num,
                            'id': sentence.id,
                        })
                elif level == 'text':
                    for text in block.texts:
                        data_points.append({
                            'level': level,
                            'document_id': document.name,
                            'text': text.text,
                            'page_num': page.num,
                            'id': text.id,
                        })

    dataset = datasets.Dataset.from_list(data_points)

    return dataset


def _annotate_document(
        document: Document,
        annotator_name: str,
        dataset: datasets.Dataset,
        level: str,
        overwrite: bool = True,
) -> Document:
    if level not in ['page', 'block', 'sentence']:
        raise ValueError(f'The specified level ({level}) is invalid. It must be either page, block, or sentence.')

    document = copy.deepcopy(document)

    if overwrite:
        for annot_obj, _ in document.find_all_annotations_by_annotator_name(annotator_name=annotator_name):
            annot_obj.remove_annotations_by_annotator_name(annotator_name=annotator_name)

    for d in dataset:

        page = document.find_page_by_num(d['page_num'])

        if level == 'page':
            parent_object = page
        elif level == 'block':
            block = page.find_block_by_id(d['id'])
            parent_object = block
        elif level == 'sentence':
            sentence = page.find_sentence_by_id(d['id'])
            parent_object = sentence
        else:
            raise NotImplementedError

        annot_obj = Annotation(
            parent_object=parent_object,
            annotator=annotator_name,
            value=d['label'],
            meta={'score': d['score']}
        )
        parent_object.add_annotation(annot_obj)

    return document


def annotate_by_sequence_classification(
    annotator_name: str,
    document: Document,
    tokenizer: PreTrainedTokenizerFast,
    model: AutoModelForSequenceClassification,
    level: str,
    target_layouts: List[str] = None,
    batch_size: int = 1,
    multi_label: bool = False,
) -> Document:
    if level not in LEVEL_NAMES:
        raise ValueError(f'The specified level ({level}) is invalid. It must be one of {LEVEL_NAMES}.')
    unk_layouts = set(target_layouts) - LAYOUT_NAMES
    if unk_layouts:
        raise ValueError(f'The specified target_layouts ({unk_layouts}) are invalid. '
                         f'It must be either title, list, text, or cell.')

    dataset = _load_dataset(document=document, level=level, target_layouts=target_layouts)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
        top_k=None if multi_label else 1
    )

    # For the single-label classification
    labels = []
    scores = []
    # For the multi-label classification
    label_name_2_labels = {l: [] for l in pipe.model.config.label2id.keys()}
    label_name_2_scores = {l: [] for l in pipe.model.config.label2id.keys()}

    for out in tqdm(
            pipe(KeyDataset(dataset, "text"), batch_size=batch_size, truncation="only_first"),
            total=len(dataset)
    ):
        if not multi_label:
            labels.append(out[0]['label'])
            scores.append(out[0]['score'])
        else:
            for o in out:
                label_name_2_labels[o['label']].append('yes' if o['score'] >= 0.5 else 'no')
                label_name_2_scores[o['label']].append(o['score'])

    if not multi_label:
        assert len(labels) == len(scores) == len(dataset)
        annotated_dataset = dataset.add_column('label', labels).add_column('score', scores)
        document = _annotate_document(
            document=document,
            annotator_name=annotator_name,
            dataset=annotated_dataset,
            level=level
        )
    else:
        for label_name in pipe.model.config.label2id.keys():
            labels = label_name_2_labels[label_name]
            scores = label_name_2_scores[label_name]
            assert len(labels) == len(scores) == len(dataset)
            annotated_dataset = dataset.add_column('label', labels).add_column('score', scores)
            document = _annotate_document(
                document=document,
                annotator_name=annotator_name + f'_{label_name}',
                dataset=annotated_dataset,
                level=level
            )

    return document

