from logging import getLogger
import traceback
import argparse
import os
from distutils.util import strtobool

from reportparse.structure.document import Document
from reportparse.reader.base import BaseReader
from reportparse.annotator.base import BaseAnnotator
from reportparse.util.helper import HFModelCache


def main(args):
    """
    The main entry point of ReportParse.
    """

    logger = getLogger(__name__)
    logger.info(f'Arguments: {args}')

    if len(args.annotators) == 0:
        logger.warning(f'You do not specify any annotators. You will only get the document layout analysis results.')

    # Load PDF files
    input_file_paths = []
    if os.path.isdir(args.input):
        logger.info(f'You specify a directory as the input. '
                    f'We will find and analyze all {args.input_type} files under it.')
        for basename in sorted(os.listdir(args.input)):
            fpath = os.path.join(args.input, basename)
            fpath = os.path.realpath(fpath)
            if os.path.isfile(fpath):
                if args.input_type == 'pdf' and fpath.lower().endswith('.pdf'):
                    input_file_paths.append(fpath)
                elif args.input_type == 'json' and fpath.lower().endswith('.pdf.json'):
                    input_file_paths.append(fpath)
    else:
        if not os.path.isfile(args.input):
            raise FileNotFoundError(
                f"The input file ({args.input}) is not a valid file."
            )
        if (args.input.lower().endswith('.pdf') and args.input_type != 'pdf') or \
                (args.input.lower().endswith('.pdf.json') and args.input_type != 'json'):
            raise ValueError(
                f"The input_type ({args.input_type}) is not consistent with the input file ({args.input})"
                "Use --overwrite_output_dir to overcome."
            )
        elif not (args.input.lower().endswith('.pdf') or args.input.lower().endswith('.pdf.json')):
            raise ValueError(
                f"The extension of the input file ({args.input}) is not valid. Please specify the PDF or JSON file."
            )
        input_file_paths.append(args.input)

    logger.info(f'We will analyze the following {args.input_type} file(s): {input_file_paths}')

    if not os.path.exists(args.output_dir) or not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)

    # Prepare the PDF reader
    reader = BaseReader.by_name(args.reader)()

    # Prepare the annotator
    annotators = []
    for annotator_name in args.annotators:
        annotator = BaseAnnotator.by_name(annotator_name)()
        annotators.append(annotator)

    # Process the input PDF files
    for input_path in input_file_paths:
        logger.info(f'###### Processing "{input_path}". ######')

        if args.input_type == 'pdf':
            output_path = os.path.join(args.output_dir, os.path.basename(input_path) + '.json')
        else:
            output_path = os.path.join(args.output_dir, os.path.basename(input_path))

        try:
            document: Document = None

            if os.path.exists(output_path):
                if args.overwrite_strategy == "no":
                    logger.warning(f'The output file "{output_path}" already exists. Skip this file.')
                    continue
                elif args.overwrite_strategy == "all":
                    logger.warning(f'The existing output file "{output_path}" will be '
                                   f'replaced completely by the new result.')
                    if args.input_type == 'json':
                        raise RuntimeError('You can not use the overwrite_strategy "all" for "json" input '
                                           'because "reader" requires "pdf" input. '
                                           'Use "annotation-clear" or "annotation-add" instead.')
                elif args.overwrite_strategy == "annotator-clear":
                    logger.warning(f'The output file "{output_path}" exists. '
                                   f'We will use the existing "reader" result, '
                                   f'clear all the existing annotator results, add new annotator results.')
                    logger.info(f'Load document data from "{output_path}"')
                    logger.warning(f'We ignore "reader" ({reader}) '
                                   f'because you are loading document layouts from the json file.')
                    document = Document.from_json_file(file_path=output_path)
                    logger.info(f'Clear all the existing annotations')
                    document.remove_all_annotations()
                elif args.overwrite_strategy == "annotator-add":
                    logger.warning(f'The output file "{output_path}" exists. '
                                   f'We will use the existing "reader" result and add new annotator results.')
                    logger.info(f'Load document data from "{output_path}"')
                    logger.warning(f'We ignore "reader" ({reader}) '
                                   f'because you are loading document layouts from the json file.')
                    document = Document.from_json_file(file_path=output_path)

            # Read the PDF or json file
            if document is None:
                if args.input_type == 'json':
                    logger.warning(f'We ignore "reader" ({reader}) '
                                   f'because you are loading document layouts from the json file.')
                    document = Document.from_json_file(file_path=input_path)
                else:
                    logger.info(f'Apply the reader "{reader}"')
                    document = reader.read(input_path=input_path, args=args)

            # Apply annotators
            for annotator in annotators:
                logger.info(f'Apply the annotator "{annotator}".')
                document = annotator.annotate(document=document, args=args)
                logger.info(f'Finished annotations by "{annotator}".')
                logger.info(f'Current cached huggingface models: {HFModelCache().current_model_cache.keys()}')
                logger.info(f'Current cached huggingface tokenizers: {HFModelCache().current_tokenizer_cache.keys()}')

            # Save the easy-to-use CSV datasets
            if args.output_csv_dataset:
                for level in ['page', 'block', 'sentence', 'table', 'figure']:
                    if args.input_type == 'pdf':
                        output_csv_path = os.path.join(
                            args.output_dir, os.path.basename(input_path) + f'.{level}-level-dataset.csv')
                    else:
                        output_csv_path = os.path.join(
                            args.output_dir, os.path.basename(input_path)[:-len('.json')] + f'.{level}-level-dataset.csv')
                    document.to_dataframe(level=level).to_csv(output_csv_path, header=True, index=False)
                    logger.info(f'Saved the dataset output file to: "{output_csv_path}".')

            # Save the Document object as a JSON file
            document.save(output_path)
            logger.info(f'Saved the full output file to: "{output_path}".')

            logger.info(f'Completed. See the output files under "{args.output_dir}".')

        except BaseException as e:
            logger.error(f'Failed to analyze {input_path} (see the error message below).')
            logger.error(traceback.format_exc())

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i',
        '--input',
        type=str,
        help='The input file or directory path',
        default=None,
        required=True
    )
    parser.add_argument(
        '-o',
        '--output_dir',
        type=str,
        help='The output directory path',
        default='./',
    )
    parser.add_argument(
        '--input_type',
        type=str,
        choices=['pdf', 'json'],
        help='The input file type. "pdf" is a PDF file. '
             '"json" is the output file of ReportParse where we will load data only from it.',
        default='pdf',
    )
    parser.add_argument(
        '--reader',
        type=str,
        choices=BaseReader.list_available(),
        help='The name of the PDF layout / text extraction method',
        default='pymupdf',
    )
    parser.add_argument(
        '--annotators',
        type=str,
        nargs='+',
        choices=BaseAnnotator.list_available(),
        help='The annotation methods to apply',
        default=[],
    )
    parser.add_argument(
        '--max_pages',
        type=int,
        help='The number of max pages to load by the reader',
        default=None,
    )
    parser.add_argument(
        '--skip_pages',
        type=int,
        nargs='+',
        help='The pages to skip. Zero-indexed.',
        default=[],
    )
    parser.add_argument(
        '--skip_load_image',
        type=strtobool,
        help='Whether to skip loading the image of pages.',
        default=False,
    )
    parser.add_argument(
        '--overwrite_strategy',
        type=str,
        choices=['no', 'all', 'annotator-clear', 'annotator-add'],
        help='Whether to overwrite the output file if it exists. '
             '"no" will not overwrite the output file.'
             '"all" will replace the existing output file with the completely new one.'
             '"annotator-clear" will use existing "reader" results but does not use old annotator results.'
             '"annotator-add" will use existing "reader" results and overwrite the annotator results '
             'only for the specified annotators.',
        default='no'
    )
    parser.add_argument(
        '--output_csv_dataset',
        type=strtobool,
        help='Whether to output the csv datasets',
        default=True,
    )

    for reader_cls_name in BaseReader.list_available():
        reader_cls = BaseReader.by_name(reader_cls_name)()
        reader_cls.add_argument(parser)

    for annot_cls_name in BaseAnnotator.list_available():
        annot_cls = BaseAnnotator.by_name(annot_cls_name)()
        annot_cls.add_argument(parser)

    args = parser.parse_args()
    main(args)

