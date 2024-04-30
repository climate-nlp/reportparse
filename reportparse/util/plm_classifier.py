import copy
from typing import List
from tqdm.auto import tqdm

import torch
from transformers import AutoModelForSequenceClassification, PreTrainedTokenizerFast, pipeline
from transformers.pipelines.pt_utils import KeyDataset
import datasets

from reportparse.structure.document import Document, Annotation


def _load_dataset(document: Document, level: str, target_layouts: List[str] = None) -> datasets.Dataset:
    if level not in ['page', 'block', 'sentence']:
        raise ValueError(f'The specified level ({level}) is invalid. It must be either page, block, or sentence.')

    data_points = []

    for page in document.pages:

        if level == 'page':
            data_points.append({
                'document_id': document.name,
                'text': page.text,
                'page_num': page.num,
            })

        elif level == 'block':

            for block in page.blocks:
                if target_layouts is not None and block.layout_type not in target_layouts:
                    continue
                data_points.append({
                    'document_id': document.name,
                    'text': block.text,
                    'page_num': page.num,
                    'id': block.id,
                })

        elif level == 'sentence':

            for block in page.blocks:
                if target_layouts is not None and block.layout_type not in target_layouts:
                    continue
                for sentence in block.sentences:
                    data_points.append({
                        'document_id': document.name,
                        'text': sentence.text,
                        'page_num': page.num,
                        'id': sentence.id,
                    })

    dataset = datasets.Dataset.from_list(data_points)

    return dataset


def _annotate_document(
        document: Document,
        annotator_name: str,
        dataset: datasets.Dataset,
        level: str,
        overwrite: bool = True
) -> Document:
    if level not in ['page', 'block', 'sentence']:
        raise ValueError(f'The specified level ({level}) is invalid. It must be either page, block, or sentence.')

    if overwrite:
        for annot_obj, _ in document.find_annotations_by_annotator_name(annotator_name=annotator_name):
            annot_obj.remove_annotator(annotator_name=annotator_name)

    document = copy.deepcopy(document)

    for d in dataset:

        page = document.find_page_by_num(d['page_num'])

        if level == 'page':
            page.add_annotation(
                Annotation(
                    parent_object=page, annotator=annotator_name, value=d['label'], meta={'score': d['score']})
            )

        elif level == 'block':
            block = page.find_block_by_id(d['id'])
            block.add_annotation(
                Annotation(
                    parent_object=block, annotator=annotator_name, value=d['label'], meta={'score': d['score']})
            )

        elif level == 'sentence':
            sentence = page.find_sentence_by_id(d['id'])
            sentence.add_annotation(
                Annotation(
                    parent_object=sentence, annotator=annotator_name, value=d['label'], meta={'score': d['score']})
            )

    return document


def annotate_by_sequence_classification(
    annotator_name: str,
    document: Document,
    tokenizer: PreTrainedTokenizerFast,
    model: AutoModelForSequenceClassification,
    level: str,
    target_layouts: List[str] = None,
    batch_size: int = 1,
) -> Document:
    if level not in ['page', 'block', 'sentence']:
        raise ValueError(f'The specified level ({level}) is invalid. It must be either page, block, or sentence.')

    dataset = _load_dataset(document=document, level=level, target_layouts=target_layouts)

    pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1
    )

    labels = []
    scores = []
    for out in tqdm(
            pipe(KeyDataset(dataset, "text"), batch_size=batch_size, truncation="only_first"),
            total=len(dataset)
    ):
        labels.append(out['label'])
        scores.append(out['score'])

    assert len(labels) == len(scores) == len(dataset)
    dataset = dataset.add_column('label', labels)
    dataset = dataset.add_column('score', scores)

    document = _annotate_document(
        document=document,
        annotator_name=annotator_name,
        dataset=dataset,
        level=level
    )

    return document

