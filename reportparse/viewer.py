import os
import argparse
import copy

import pandas as pd
import uuid
import gradio as gr
import spacy
import fitz
from wordcloud import WordCloud

from reportparse.structure.document import Document
from reportparse.annotator.base import BaseAnnotator
from reportparse.reader.pymupdf_reader import load_image_from_page, load_dummy_page_image
from reportparse.util.helper import draw_layout_on_page


MAX_IMAGES = 5

config_overwrite = [
    "USE_LAYOUT=False",
    "USE_PDF_MINER=False",
    "USE_TABLE_SEGMENTATION=False",
    "USE_TABLE_REFINEMENT=False",
    "USE_OCR=False"
]
#analyzer = dd.get_dd_analyzer(config_overwrite=config_overwrite)

nlp = spacy.load("en_core_web_sm")

path_2_cache = dict()

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, f'log{os.path.sep}viewer'))
os.makedirs(DATA_DIR, exist_ok=True)
os.environ['GRADIO_TEMP_DIR'] = DATA_DIR

ASSET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), f'asset'))


def render(
    input_json_basename: str, json_dir: str, pdf_dir: str,
    load_image: bool,
    show_block: bool, block_color: str,
    show_table_block: bool, table_block_color: str,
    show_figure_block: bool, figure_block_color: str,
    show_sentence: bool, sentence_color: str,
    show_word: bool, word_color: str,
    annotator_name: str, prob_threshold: float, word_threshold: int, label_color: str,
    progress=gr.Progress()
):
    #global analyzer
    global DATA_DIR
    global path_2_cache
    global MAX_IMAGES

    input_json_file = os.path.join(json_dir, input_json_basename)

    document: Document = Document.from_json_file(file_path=input_json_file)

    annots = []
    basic_data = []
    for page in progress.tqdm(document.pages, desc='Preparing annotation stats'):

        basic_data.append({'page': page.num, 'type': 'block', 'count': len(page.blocks)})
        basic_data.append({'page': page.num, 'type': 'table', 'count': len(page.tables)})
        basic_data.append({'page': page.num, 'type': 'figure', 'count': len(page.figures)})
        #n_sentences = len([s for b in page.blocks for s in b.sentences])
        #basic_data.append({'page': page.num, 'type': 'sentence', 'count': n_sentences})

        for annot_obj, annot in page.find_all_annotations_by_annotator_name(annotator_name):

            score = annot.meta['score'] if isinstance(annot.meta, dict) and 'score' in annot.meta else 1
            if score < prob_threshold:
                continue

            text = annot_obj.text
            if annotator_name != 'climate_figure' and len(nlp(text)) < word_threshold:
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
            if not text.strip():
                continue
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

    progress(0.0, desc="Loading PDF files (will take several minutes if the PDF contains multiple pages)")

    pdf_path = os.path.join(
        pdf_dir,
        input_json_basename[:-len('.json')]
    )

    cache_key = (input_json_file, load_image)
    if input_json_file in path_2_cache:
        orig_images, orig_page_nums = path_2_cache[cache_key]
    else:
        try:
            orig_images = []
            orig_page_nums = []
            doc = fitz.open(pdf_path)
            for page in doc:
                if load_image:
                    image_info = load_image_from_page(page=page)
                else:
                    image_info = load_dummy_page_image(page=page)
                img = image_info['img']
                if len(orig_images) >= MAX_IMAGES:
                    break
                orig_images.append(img)
                orig_page_nums.append(page.number)
                path_2_cache[cache_key] = (orig_images, orig_page_nums)
        except BaseException as e:
            print(e)
            raise gr.Error(f"Could not read the PDF file. Try another file.")

    layout_images = []

    for orig_image, page_num in progress.tqdm(zip(orig_images, orig_page_nums), desc='Drawing output images'):
        page = document.find_page_by_num(page_num=page_num)
        page.image = copy.deepcopy(orig_image)
        img = draw_layout_on_page(
            page=page,
            show_annotation=True,
            annotator_name=annotator_name,
            annotation_color=label_color,
            show_block=show_block,
            block_color=block_color,
            show_table_block=show_table_block,
            table_block_color=table_block_color,
            show_figure_block=show_figure_block,
            figure_block_color=figure_block_color,
            show_sentence=show_sentence,
            sentence_color=sentence_color,
            show_word=show_word,
            word_color=word_color,
        )
        layout_images.append((img, f'Page {page.num + 1}'))

    save_dir = os.path.join(DATA_DIR, str(uuid.uuid4()) + os.sep)
    os.makedirs(save_dir)

    summary_text = f"""- Selected file: {input_json_basename}
- Download PDF file: [{os.path.basename(pdf_path)}](/file={pdf_path})
- Download full data: [{os.path.basename(input_json_file)}](/file={input_json_file})
"""

    if annots is not None:
        csv_save_path = os.path.join(
            save_dir, f'{os.path.basename(input_json_file)}.{annotator_name}_wrd-thld-{word_threshold}_prb-thld-{prob_threshold}.csv'
        )
        annots.to_csv(csv_save_path, index=False)
        summary_text += f"- Download annotation data: [{os.path.basename(csv_save_path)}](/file={csv_save_path})"

    progress(1.0, desc="Sending output results")

    return summary_text, annots, basic_plot, annot_plot, layout_images, wordcloud_images


def main(args):
    global MAX_IMAGES
    global ASSET_DIR

    json_paths = []
    for basename in os.listdir(args.json_dir):
        fpath = os.path.join(args.json_dir, basename)
        if os.path.isfile(fpath) and fpath.endswith('.json'):
            json_paths.append(fpath)
    json_paths = sorted(json_paths)
    json_base_names = [os.path.basename(jp) for jp in json_paths]

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
                gr.Markdown("""# [analyzed result viewer]
Report Parse is a tool to analyze layout of corporate responsibility reports (e.g., ESG and sustainability reports), apply off-the-shelf NLP models to annotate the report, and visualize the results.
- Note that this viewer <span style="color:red">visualizes the already analyzed results (dumped in JSON files) only</span>. So instead of uploading PDF files, you can only see the visualization of the analyzed result.
    - To use the real-time analysis demo of PDF files, use reportparse.demo instead.
    - To use a command-line tool to analyze PDF files, use reportparse.main.""")
            with gr.Column(scale=1):
                gr.Image(f'{os.path.join(ASSET_DIR, "reportparse_overview.png")}', show_label=False)

        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(variant="panel", scale=1):
                    gr.Markdown("## 1. Choose input file")
                    input_json_basename = gr.Dropdown(
                        json_base_names, label="JSON file",
                        info="Choose the PDF analyzed file",
                        value=json_base_names[0],
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
                                show_table_block = gr.Checkbox(
                                    label="Table block",
                                    info="",
                                    min_width=5, value=True, scale=2
                                )
                                table_block_color = gr.ColorPicker(
                                    label="",
                                    show_label=False,
                                    value="#2f73a1",
                                    info='Bounding box color',
                                    min_width=5,
                                    scale=1,
                                )
                        with gr.Box():
                            with gr.Group():
                                show_figure_block = gr.Checkbox(
                                    label="Figure",
                                    info="",
                                    min_width=5, value=True, scale=2
                                )
                                figure_block_color = gr.ColorPicker(
                                    label="",
                                    show_label=False,
                                    value="#ff7321",
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
                        gr.Markdown(f"##### Layout of first {MAX_IMAGES} pages")
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
                input_json_basename, gr.State(args.json_dir), gr.State(os.path.abspath(args.pdf_dir)),
                load_image,
                show_block, block_color,
                show_table_block, table_block_color,
                show_figure_block, figure_block_color,
                show_sentence, sentence_color,
                show_word, word_color,
                annotator_name, prob_threshold, word_threshold, label_color
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
        allowed_paths=[DATA_DIR, ASSET_DIR, os.path.abspath(args.pdf_dir), os.path.abspath(args.json_dir)],
        favicon_path=os.path.join(ASSET_DIR, 'reportparse.png')
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pdf_dir',
        type=str,
        help='The input PDF file directory path',
        default=None,
        required=True
    )
    parser.add_argument(
        '--json_dir',
        type=str,
        help='The input json file directory path',
        default=None,
        required=True
    )
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
        '--annotators',
        type=str,
        nargs='+',
        choices=[a for a in BaseAnnotator.list_available() if a != "custom_huggingface"],
        help='The annotation methods to apply',
        default=[],
    )
    args = parser.parse_args()
    main(args)

