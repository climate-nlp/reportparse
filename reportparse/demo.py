import os
import argparse
from typing import Optional, List, Dict, Tuple
import numpy as np
import pandas as pd
import hashlib
import deepdoctection as dd
import gradio as gr
import spacy
import copy
import cv2
import json
from PIL import ImageColor
from wordcloud import WordCloud

from reportparse.annotator.base import BaseAnnotator
from reportparse.reader.base import BaseReader


nlp = spacy.load("en_core_web_sm")

filehash_2_document_cache = dict()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, f'log{os.path.sep}demo'))
os.makedirs(DATA_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = DATA_DIR

ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), f'asset'))


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


def render(
    input_pdf_file, max_pages: int,
    load_image: bool,
    show_block: bool, block_color: str,
    show_sentence: bool, sentence_color: str,
    show_word: bool, word_color: str,
    reader_name: str, annotator_name: str, prob_threshold: float, word_threshold: int, label_color: str,
    progress=gr.Progress()
):
    global filehash_2_document_cache

    filehash = hashlib.md5(open(input_pdf_file.name, 'rb').read()).hexdigest()

    cache_key = (
        filehash, max_pages, reader_name, annotator_name, load_image
    )
    if cache_key not in filehash_2_document_cache:
        progress(0.1, 'Loading PDF layouts (will take a significant time if you upload a PDF file with many pages)')
        try:
            reader = BaseReader.by_name(reader_name)()
            document = reader.read(
                input_path=input_pdf_file.name,
                max_pages=max_pages,
                skip_pages=None,
                skip_load_image=not load_image,
            )
        except BaseException as e:
            print(e)
            raise gr.Error(f"Could not read the PDF file. Try another file.")

        progress(0.5, f'Annotating text by "{annotator_name}"')
        annotator = BaseAnnotator.by_name(annotator_name)()
        document = annotator.annotate(document=document, args=None)
        filehash_2_document_cache[cache_key] = document
    else:
        document = filehash_2_document_cache[cache_key]

    annots = []
    basic_data = []
    for page in progress.tqdm(document.pages, desc='Preparing annotation stats'):

        basic_data.append({'page': page.num, 'type': 'block', 'count': len(page.blocks)})
        n_sentences = len([s for b in page.blocks for s in b.sentences])
        basic_data.append({'page': page.num, 'type': 'sentence', 'count': n_sentences})

        for annot_obj, annot in page.find_annotations_by_annotator_name(annotator_name):

            score = annot.meta['score'] if isinstance(annot.meta, dict) and 'score' in annot.meta else 1
            if score < prob_threshold:
                continue

            text = annot_obj.text
            if len(nlp(text)) < word_threshold:
                continue

            annots.append({
                'id': annot.id,
                'page': page.num + 1,
                'annotator': annot.annotator,
                'value': annot.value,
                'score': f'{score:.3f}',
                'text': text,
            })

    progress(0.0, desc="Generating plots")

    wordcloud_images = []

    if annots:
        annots = pd.DataFrame(annots)
        annot_plot = gr.BarPlot(
            annots['value'].value_counts().reset_index(),
            x="value",
            y="count",
            vertical=False,
            width=300,
            height=200,
            color="value",
            title="The number of labels",
            interactive=True,
        )

        annot_labels = annots['value'].unique()
        annot_labels = sorted(annot_labels)
        for annot_label in annot_labels:
            text = annots[annots['value'] == annot_label]['text']
            text = ' '.join(text)
            wc_img = WordCloud(
                background_color='white',
                width=500,
                height=150
            ).generate(text).to_image()
            wordcloud_images.append((wc_img, annot_label))
    else:
        annots = None
        annot_plot = None

    if basic_data:
        basic_plot = gr.LinePlot(
            pd.DataFrame(basic_data),
            x="page",
            y="count",
            color="type",
            title="Document stats",
            width=300,
            height=70,
        )
    else:
        basic_plot = None

    layout_images = []

    for page in progress.tqdm(document.pages[:5], desc='Drawing output images'):

        img = copy.deepcopy(page.image)

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

        if show_sentence:
            box_stack = []
            category_names_list = []
            for block in page.blocks:
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
            for block in page.blocks:
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

        # Semantic labels
        box_stack = []
        category_names_list = []
        for annot_obj, annot in page.find_annotations_by_annotator_name(annotator_name):

            score = annot.meta['score'] if isinstance(annot.meta, dict) and 'score' in annot.meta else 1
            if score < prob_threshold:
                continue

            text = annot_obj.text
            if len(nlp(text)) < word_threshold:
                continue

            if hasattr(annot_obj, 'bbox'):
                box_stack.append(annot_obj.bbox)
                category_names_list.append(annot.value)

        if box_stack:
            boxes = np.vstack(box_stack)
            img = draw_boxes(
                np_image=img,
                boxes=boxes,
                category_names_list=category_names_list,
                category_to_color={k: ImageColor.getcolor(label_color, "RGB") for k in set(category_names_list)},
                font_scale=1.5,
                rectangle_thickness=6,
            )

        layout_images.append((img, f'Page {page.num + 1}'))

    data_dir = os.path.dirname(input_pdf_file.name)
    json_save_path = os.path.join(data_dir, f'{os.path.basename(input_pdf_file.name)}.json')
    with open(json_save_path, 'w') as f:
        f.write(json.dumps(document.to_dict(), ensure_ascii=False, indent=4))
    summary_text = f"- Download full data: [{os.path.basename(json_save_path)}](/file={json_save_path})\n"

    if annots is not None:
        csv_save_path = os.path.join(
            data_dir, f'{os.path.basename(input_pdf_file.name)}.{reader_name}_{annotator_name}_wrd-thld-{word_threshold}_prb-thld-{prob_threshold}.csv'
        )
        annots.to_csv(csv_save_path, index=False)
        summary_text += f"- Download annotation data: [{os.path.basename(csv_save_path)}](/file={csv_save_path})"

    progress(1.0, desc="Sending output results")

    return summary_text, annots, basic_plot, annot_plot, layout_images, wordcloud_images


def main(args):
    global ASSET_DIR

    if args.readers:
        reader_names = args.readers
    else:
        reader_names = BaseReader.list_available()
    reader_names = sorted(reader_names)

    if args.annotators:
        annotator_names = args.annotators
    else:
        annotator_names = [a for a in BaseAnnotator.list_available() if a != "custom_huggingface"]
    annotator_names = sorted(annotator_names)

    demo = gr.Blocks(theme=gr.themes.Soft(), analytics_enabled=False, title='ReportParse')

    with demo:
        gr.HTML(f'<img src="/file={os.path.join(ASSET_DIR, "reportparse.png")}" width="200">')

        with gr.Row():
            with gr.Column(scale=3):
                gr.Markdown("""# [analyzer]
Report Parse is a tool to analyze layout of corporate responsibility reports (e.g., ESG and sustainability reports), apply off-the-shelf NLP models to annotate the report, and visualize the results.
- To use a offline demo of PDF files (i.e., the viewer of already analyzed results dumped by JSON files), use reportparse.viewer instead.
- To use a command-line tool to analyze PDF files, use reportparse.main.""")
            with gr.Column(scale=1):
                gr.Image(f'{os.path.join(ASSET_DIR, "reportparse_overview.png")}', show_label=False)

        with gr.Row():

            with gr.Column(scale=4):
                with gr.Column(variant="panel", scale=1):
                    gr.Markdown("## 1. Choose input file")
                    input_pdf_file = gr.File(
                        label="An ESG/sustainability report (PDF) file",
                        show_label=True,
                        file_types=['.pdf'],
                        file_count='single',
                        type='file'
                    )
                    max_pages = gr.Slider(
                        1, 200, value=5, step=1,
                        label="Max pages",
                        info="Will truncate the results based on the page num",
                    )
                    gr.Markdown(
                        '<span style="color:red">Larger "max pages" will take significant time to analyze.</span>. '
                        'Try a small value (e.g., 5) at first to test and then try larger value.'
                    )

                with gr.Column(variant='compact', scale=1):
                    gr.Markdown("## 2. Layout visualization setting")
                    load_image = gr.Checkbox(
                        value=True,
                        label="Display PDF images"
                    )
                    with gr.Row():
                        with gr.Box():
                            with gr.Group():
                                show_block = gr.Checkbox(
                                    label="Block",
                                    info="",
                                    min_width=5, value=True, scale=2
                                )
                                block_color = gr.ColorPicker(
                                    label="",
                                    show_label=False,
                                    value="#ff73a1",
                                    info='Bounding box color',
                                    min_width=5,
                                    scale=1,
                                )
                        with gr.Box():
                            with gr.Group():
                                show_sentence = gr.Checkbox(
                                    label="Sentence",
                                    info="",
                                    min_width=5, value=False, scale=2
                                )
                                sentence_color = gr.ColorPicker(
                                    label="",
                                    show_label=False,
                                    value="#2626ff",
                                    info='Bounding box color',
                                    min_width=5, scale=1
                                )
                        with gr.Box():
                            with gr.Group():
                                show_word = gr.Checkbox(
                                    label="Text box",
                                    info="",
                                    min_width=5, value=False, scale=2
                                )
                                word_color = gr.ColorPicker(
                                    label="",
                                    show_label=False,
                                    value="#8a8a8a",
                                    info='Bounding box color',
                                    min_width=5, scale=1
                                )

            with gr.Column(variant='compact', scale=3):
                gr.Markdown("## 3. Annotation visualization setting")
                gr.Markdown("Visually display the annotations by each method (e.g., language models) for either blocks, sentences, or text boxes.")
                reader_name = gr.Dropdown(
                    reader_names, label="Reader method", info="", value=reader_names[0],
                )
                annotator_name = gr.Dropdown(
                    annotator_names, label="Annotation method", info="", value=annotator_names[0],
                )
                with gr.Group():
                    with gr.Row():
                        prob_threshold = gr.Slider(
                            0, 1, value=0.5, step=0.05,
                            label="Score threshold",
                            info="Will truncate the results based on the model's probability score",
                            scale=2
                        )
                        word_threshold = gr.Slider(
                            1, 100, value=20, step=1,
                            label="Word threshold",
                            info="Will truncate the results based on the num. of minimum words",
                            scale=2
                        )
                        label_color = gr.ColorPicker(
                            label="",
                            show_label=False,
                            info='Bounding box color',
                            value="#d21cff",
                            scale=1
                        )

        btn = gr.Button("Show results", variant="primary", size='lg', elem_id='show_button')

        with gr.Column():
            gr.Markdown("# Outputs", elem_id='output_title')
            summary_text = gr.Markdown('')

        with gr.Box():
            gr.Markdown('## Summary')
            with gr.Row():
                with gr.Column(variant='panel', scale=5):
                    with gr.Box():
                        gr.Markdown(f"##### Layout of first 5 pages")
                        layout_gallery = gr.Gallery(
                            label=f"", show_label=False, elem_id="layout_gallery",
                            columns=1, rows=1, height=700, preview=True, object_fit='scale-down',
                        )
                with gr.Column(variant='panel', scale=3):
                    with gr.Box():
                        basic_plot = gr.LinePlot(label='Layout stats')
                    with gr.Box():
                        annotation_plot = gr.BarPlot(label='Annotation stats')
                    with gr.Box():
                        gr.Markdown(f"##### Wordcloud for each label")
                        wordcloud_gallery = gr.Gallery(
                            label="", show_label=False, elem_id="wordcloud_gallery",
                            columns=1, rows=1, height=200, preview=True, object_fit='scale-down',
                        )

        with gr.Box():
            gr.Markdown('## Full data of the annotation method')
            with gr.Column():
                annotation_data = gr.Dataframe(wrap=True, height=1000, interactive=True)

        btn.click(
            fn=render,
            inputs=[
                input_pdf_file, max_pages,
                load_image,
                show_block, block_color,
                show_sentence, sentence_color,
                show_word, word_color,
                reader_name, annotator_name, prob_threshold, word_threshold, label_color
            ],
            outputs=[summary_text, annotation_data, basic_plot, annotation_plot, layout_gallery, wordcloud_gallery],
        )

        demo.load(_js="""
        function scroll_to_output() {
          const button = document.querySelector("#show_button");
          button.addEventListener("click", e => {
                document.getElementById('output_title').scrollIntoView();
          });
        }
        """)

    demo.queue(max_size=10).launch(
        share=False,
        debug=False,
        server_name=args.server_name,
        server_port=args.server_port,
        ssl_verify=False,
        allowed_paths=[DATA_DIR, ASSET_DIR],
        favicon_path=os.path.join(ASSET_DIR, 'reportparse.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--server_name',
        type=str,
        help='The host name',
        default=None,
    )
    parser.add_argument(
        '--server_port',
        type=int,
        help='The port number',
        default=None,
    )
    parser.add_argument(
        '--readers',
        type=str,
        nargs='+',
        choices=BaseReader.list_available(),
        help='The selectable reader methods',
        default=[],
    )
    parser.add_argument(
        '--annotators',
        type=str,
        nargs='+',
        choices=[a for a in BaseAnnotator.list_available() if a != "custom_huggingface"],
        help='The selectable annotation methods',
        default=[],
    )

    for annot_cls_name in [a for a in BaseAnnotator.list_available() if a != "custom_huggingface"]:
        annot_cls = BaseAnnotator.by_name(annot_cls_name)()
        annot_cls.add_argument(parser)

    args = parser.parse_args()
    main(args)

