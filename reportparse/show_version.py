import subprocess

import PyPDF2
import deepdoctection
import fitz
import PIL
import spacy
import torch
import transformers
import numpy
import pandas
import gradio
import cv2


def main():
    """
    The entrypoint to check package versions.
    """
    print('The following shows installed packages.')
    print('If you face errors, the packages may not be installed or may be broken.')
    print('Note that we recommend using the versions: qpdf-11.6.4, poppler tesseract-5.3.0')
    print()

    print('================== ReportParse version ==================')
    try:
        commit_id = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode('ascii').strip()
        print(f'Commit ID: {commit_id}')
    except BaseException as e:
        print(e)
    print()

    print('======== Reader-related python package versions =========')
    print(f'deepdoctection: {deepdoctection.__version__}')
    print(f'PyPDF2: {PyPDF2.__version__}')
    print(f'fitz: {fitz.__version__}')
    print(f'PIL: {PIL.__version__}')
    print(f'spacy: {spacy.__version__}')
    print()

    print('======= Annotator-related python package versions ========')
    print(f'torch: {torch.__version__}')
    print(f'transformers: {transformers.__version__}')
    print(f'pandas: {pandas.__version__}')
    print(f'numpy: {numpy.__version__}')
    print()

    print('===== Visualization-related python package versions =====')
    print(f'cv2: {cv2.getVersionString()}')
    print(f'gradio: {gradio.__version__}')
    print()

    print('=========== External libs for deepdoctection ============')
    print('#### QPDF ####')
    try:
        print(subprocess.check_output(['qpdf', '--version']).decode('ascii').strip())
    except BaseException as e:
        print(f'QPDF seems not be installed.\nErrors:\n{e}')
    print()
    print('#### Poppler ####')
    try:
        print(subprocess.check_output(['pdftoppm', '-v']).decode('ascii').strip())
    except BaseException as e:
        print(f'Poppler seems not be installed.\nErrors:\n{e}')
    print()
    print('#### Tesseract ####')
    try:
        print(subprocess.check_output(['tesseract', '-v']).decode('ascii').strip())
    except BaseException as e:
        print(f'tesseract seems not be installed.\nErrors:\n{e}')
    print()
    print()
    print('#### Tesseract language files ####')
    try:
        installed_langs = subprocess.check_output(['tesseract', '--list-langs']).decode('ascii').strip()
        print('English installed:', 'eng' in installed_langs)
        print('All list:', installed_langs)
    except BaseException as e:
        print(f'tesseract seems not be installed.\nErrors:\n{e}')
    print()
    return


if __name__ == '__main__':
    main()

