



<h2 align="left">
    <img align="center" src="reportparse/asset/reportparse.png" width="160px" />
    A Unified NLP Analyzer for Corporate Sustainability Reports
</h2>

<p align="left">
    <a href="https://github.com/climate-nlp/reportparse/releases">
        <img alt="GitHub release" src="https://img.shields.io/github/release/climate-nlp/reportparse.svg">
    </a>
    <a href="https://climate-nlp.github.io/">
        <img alt="Documentation" src="https://img.shields.io/website/https/climate-nlp.github.io/index.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <img alt="GitHub release" src="https://img.shields.io/github/license/climate-nlp/reportparse.svg?color=blue">
</p>

ReportParse is a Python-based tool designed to parse corporate (sustainability) reports. 
It combines document structure analysis with natural language processing (NLP) models to extract sustainability-related information from the reports. 
We also provide easy-to-use web and command interfaces. 
The tool is expected to aid researchers and analysts in evaluating corporate commitment and management of sustainability efforts.

## Tutorials
- [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1SUF7aX62LOUhpp004zn8NItM_tOkCZc4?usp=sharing) Understanding setup and basic example
- [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1d9Oe0r3sJpag1e2wMWH6SItuBsQUXFB5?usp=sharing) Understanding setup and basic example (install without root permission for deepdoctection).
- [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1c82lWdv7xJkM1ef1jnI3Q9uyJVIsDPLY?usp=sharing) Analyzing sentiment of corporate sustainability reports


## Why should I use ReportParse?

- 💡ReportParse will reduce your workload to analyze corporate sustainability activities for your research. We know implementing the PDF text extraction and NLP model utilization for this purpose is painstaking. ReportParse will take these things instead of you.
- 💡ReportParse will be useful to test the robustness of your research. To improve the analytical robustness, you may want to try similar but different model or processing variants. ReportParse can easility change the PDF analysis method and NLP model. 
- 💡ReportParse will increase reproducibility of your analysis. ReportParse uses open sourced tools and methods. This will make it easier for other researchers to replicate your analysis.

You should use ReportParse for

- Investigating the number of environmental claims in a corporate report.
- Extracting claims related to GHG emission reduction targets in a corporate report.
- Investigating ESG topics included in a corporate report.

You should NOT use ReportParse for

- Fine-grained document structure analysis.
- Requiring 100% accuracy. (In fact, you will face a lot of noise produced by layout analysis and NLP models.)
- Automating some critical work because ReportParse usually contains noise and errors stem from the analysis.


## How does ReportParse work?

<p align="center">
  <img align="center" src="reportparse/asset/reportparse_overview.png" width="250px" />
</p>

We provide the core engine and interfaces.
Conceptually, the core engine of ReportParse was inspired by [PaperMage](https://github.com/allenai/papermage), which can extract information from scientific papers.
However, different from PaperMage, ReportParse does not consider any specific document structure because corporates publish reports in very different structure.
We support existing NLP models related to climate and sustainability domain.
For interfaces, we provide the web (based on Gradio) and command line interfaces.
Read our IJCAI 2024 demonstration paper "ReportParse: A Unified NLP Tool for Extracting Document Structure and Semantics of Corporate Sustainability Reporting" for technical detail.

### Understanding document structure, reader, and annotator

ReportParse can extract document structure from a reports.
The following figure shows the important document structure levels (page, block, and sentences) represented in ReportParse.

<p align="center">
  <img align="center" src="reportparse/asset/reportparse_structure.png" width="450px" />
</p>

By using [deepdoctection](https://github.com/deepdoctection/deepdoctection), the _reader_ can analyze the document structure. 
Then, an _annotator_ of ReportParse annotates labels for each structure level (i.e., page, block, or sentence) by using cutting-edge language models.
We integrate useful and valuable third-party models for the annotators.
For example by using ```environmental_claim``` (Stammbach et al., see [annotators](#annotators)) annotator, you can extract sentences that are related to environmental claims.
You can easily change the reader and annotator, or you can create your own reader or annotator.

See current supported [readers](#readers) and [annotators](#annotators).

### Citation

```bibtex
@inproceedings{morio-etal-2024-reportparse,
  title     = {{R}eport{P}arse: A Unified NLP Tool for Extracting Document Structure and Semantics of Corporate Sustainability Reporting},
  author    = {Morio, Gaku and In, Soh Young and Yoon, Jungah and Rowlands, Harri and Manning, Christopher D.},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  pages     = {to appear},
  year      = {2024},
  note      = {Demos},
}
```


## Quick setup

### Environment

We highly recommend you to prepare the Python environment only for ReportParse because our tool depends on complicated external package versions.
At this time, we officially support the following version.

- Python 3.8.16

### Clone the project

```bash
git clone https://github.com/climate-nlp/reportparse
cd reportparse
```

### Install dependencies

Run the following commands to install required packages.

```bash
pip install pip==23.3.1 setuptools==59.5.0 cython==3.0.6 wheel==0.42.0
pip install "deepdoctection[pt]==0.26" --no-deps
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/detectron2.git@ff53992b1985b63bd3262b5a36167098e3dada02 --no-build-isolation

python -m spacy download en_core_web_sm

# Make sure that the torch and torchvision version depend on your Python version
pip install torch==1.10.1 torchvision==0.11.2
# If you use CUDA, for example:
#pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 -f https://download.pytorch.org/whl/torch_stable.html
```
**IMPORTANT: To use deepdoctection, you need to install external packages of poppler, tesseract, leptonica, and qpdf.**

```bash
sudo apt-get update
sudo apt install -y libtool poppler-utils python3-opencv tesseract-ocr qpdf
```

If you want to install the above libs without root permissions, please refer [auto_install_deepdoctection_deps.sh](auto_install_deepdoctection_deps.sh) and example notebook [<img align="center" src="https://colab.research.google.com/assets/colab-badge.svg" />](https://colab.research.google.com/drive/1d9Oe0r3sJpag1e2wMWH6SItuBsQUXFB5?usp=sharing).

After installing all of the above, you can check if the required packages are installed.

```bash
python -m reportparse.show_version
````


## Quick start

The following shows examples of analyzing example PDF file at ```reportparse/asset/example.pdf```. 
For example, we use ```pymupdf``` as the [reader](#readers) and ```environmental_claim``` and ```sst2``` (provided by DistilBERT community) as the [annotators](#annotators).

### By the python command line tool

```bash
python -m reportparse.main \
  -i ./reportparse/asset/example.pdf \
  -o ./results \
  --input_type "pdf" \
  --overwrite_strategy "no" \
  --reader "pymupdf" \
  --annotators "environmental_claim" "sst2"
```

### By the python code

```python
# Save this file as example_code.py under the root directory of the reportparse project

import os
from reportparse.reader.base import BaseReader
from reportparse.annotator.base import BaseAnnotator

reader = BaseReader.by_name('pymupdf')()
document = reader.read(input_path='./reportparse/asset/example.pdf')

document = BaseAnnotator.by_name("environmental_claim")().annotate(document=document)
document = BaseAnnotator.by_name("sst2")().annotate(document=document)

os.makedirs('./results', exist_ok=True)

# Save the full data as a JSON file
document.save('./results/example.pdf.json')
# Save the easy-to-use dataset as a CSV file
document.to_dataframe(level='sentence').to_csv('./results/example.pdf.sentence-level-dataset.csv')
```

```bash
python -m example_code
```

### How do the results look like?

- The output JSON file would be look like [example.pdf.json](reportparse%2Fasset%2Fexample_results%2Fexample.pdf.json). If you want to investigate full document structure, this file would be informative.
- The output CSV file would be look like [example.pdf.sentence-level-dataset.csv](reportparse%2Fasset%2Fexample_results%2Fexample.pdf.sentence-level-dataset.csv). This file is useful to count labels included in a document.
  - If you use different annotation levels such as block or page, you can refer [example.pdf.block-level-dataset.csv](reportparse%2Fasset%2Fexample_results%2Fexample.pdf.block-level-dataset.csv) or [example.pdf.page-level-dataset.csv](reportparse%2Fasset%2Fexample_results%2Fexample.pdf.page-level-dataset.csv).

### Example data analysis using the output file of ReportParse

Extracting environmental claims and counting them.

```python
import pandas as pd

# Read the CSV dataset file
df = pd.read_csv('reportparse/asset/example_results/example.pdf.sentence-level-dataset.csv')
# Get environmental claim sentences
df_environment = df[df['environmental_claim'] == 'yes']
# Remove "too short" sentences
df_environment = df[(df['sentence_text'].str.split().str.len() > 20)]

# Show some example text
print(df_environment[:5])
# Results -->
# 10    Hitachi identifies, evaluates, and manages cli...
# 18    Therefore, we have established COz emissions p...
# 19    We also set and manage a metric for avoided em...
# 20    We continue to reduce COz emissions generated ...
# 22    In addition, in April 2021, Hitachi, Ltd. intr...
# Name: sentence_text, dtype: object

# Show some example texts
print('The number of total sentences:', len(df))
# Result --> The number of total sentences: 158
print('The number of environmental claim sentences:', len(df_environment))
# Result --> The number of environmental claim sentences: 62
print('Environmental claim ratio [%]:', 100 * len(df_environment) / len(df))
# Result --> Environmental claim ratio [%]: 39.24050632911393
```



## Web interfaces

We provide two types of Gradio-based interfaces to better understand the output results.

The following is the example to launch a demo server. 
You can upload your own PDF file to analyze it. 

```bash
python -m reportparse.demo \
  --server_name 0.0.0.0 \
  --server_port 60233
```

The following is the example to launch a visualization server. 
You can only select already analyzed files (i.e., JSON output files). 

```bash
python -m reportparse.viewer \
  --pdf_dir ./reportparse/asset \
  --json_dir ./results \
  --server_name 0.0.0.0 \
  --server_port 60233
```


## Other tips

### Options of the command line tool

When running ```python -m reportparse.main```, you can use following options.

| Option name          | Type                                                     | Description                                                                                                                                                                                                                                                                                                                                                                                                      |
|----------------------|----------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| -i, --input          | str                                                      | The input file or directory path. If you specify a directory, we will automatically find all files under the directory. You can specify either PDF files or JSON file (i.e., the output data file of ReportParse). If you would like to input json files, you have to change the ```--input_type``` option below.                                                                                                |
| -o, --output_dir     | str                                                      | The output directory path.                                                                                                                                                                                                                                                                                                                                                                                       |
| --input_type         | str ("pdf" or "json")                                    | The input file type. The default is "pdf". If you specify "pdf", we will consider the input file as a PDF file. If "json" is specified, we consider the input file as the output file of ReportParse where we will load data only from it.                                                                                                                                                                       |
| --reader             | str ("pymupdf" or "deepdoctection")                      | The name of the PDF layout / text extraction method. The default is "pymupdf". We currently support "pymupdf" or "deepdoctection". See more detail at [Reader types](#readers).                                                                                                                                                                                                                                  |
| --annotators         | List of str                                              | The annotation methods to apply. The annotator assigns each element (sentence, block, or page) with a label. See more detail at [Annotator types](#annotators). If you do not specify anything here, the reader will only be applied (i.e., only document structure analysis will be conducted).                                                                                                                 |
| --max_pages          | int                                                      | The number of max pages to load by the reader. We read all pages by default.                                                                                                                                                                                                                                                                                                                                     |
| --skip_pages         | List of int                                              | The pages to skip. The default is None. Zero-indexed. For example, if you would like to skip the first cover page, you can specify 0.                                                                                                                                                                                                                                                                            |
| --skip_load_image    | bool (0 or 1)                                            | Whether to skip loading the image of pages. The default is 0 (False).                                                                                                                                                                                                                                                                                                                                            |
| --overwrite_strategy | str ("no", "all", "annotator-add", or "annotator-clear") | Whether to overwrite the output file if it exists. The default is "no". "no" will not overwrite the output file. "all" will replace the existing output file with the completely new one. "annotator-clear" will use existing "reader" results but does not use old annotator results. "annotator-add" will use existing "reader" results and overwrite the annotator results only for the specified annotators. |

We also provide annotator specific optional arguments. Please refer them by running ```python -m reportparse.main --help```.

<h3 id="readers">Readers</h4>

We currently support following readers.
Note that it is impossible to provide 100% accurate reader, given the diverse nature of corporate report structure.
Please use the reader that best suits your purpose.
If you would like to add more, please contribute! 

| Reader name          | Description                                                                                                                                                                                                                                                                                   | Pro                                                                                 | Con                                                                                                             |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| ```pymupdf```        | We use Fitz of [PyMuPDF](https://pymupdf.readthedocs.io/en/latest/) to extract document structure and text from a PDF file. Note that it does no use layout analysis. We only use sentence tokenization by SpaCy. **This means the block and the sentence is exactly the same meaning here.** | <ul><li>Fast</li><li>No OCR errors</li><li>Well tested</li></ul>                    | <ul><li>No layout analysis</li><li>No text extraction for image-based PDF files</li><li>Low precision</li></ul> |
| ```deepdoctection``` | We use [deepdoctection](https://github.com/deepdoctection/deepdoctection) to analyze document structure and extract text by OCR engines. The block type includes "title", "text", and "list".                                                                                                 | <ul><li>Layout analysis</li><li>Text extraction for image-based PDF files</li></ul> | <ul><li>Slow</li><li>OCR errors</li><li>Complicated installation</li><li>Low recall</li></ul>                   |



<h3 id="annotators">Annotators</h4>

This project integrates a lot of valuable third-party models for the annotators.
We currently support following annotators (note that usually the annotator integration is done by us and the original provider is not involved.)
**Do not forget to credit the original work if you use the following annotators.**
Note that many models are not trained on the sustainability report domain, so please be in mind that the output results contain many errors.
If you would like to add more, please contribute! 

| Annotator name                      | Credit               | Reference                                                                                                                                                             | License                                                                                                                 | Description                                                                                                                   | Default level |
|-------------------------------------|:---------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|---------------|
| ```climate_commitment```            | Bingler et al.       | [Huggingface](https://huggingface.co/climatebert/distilroberta-base-climate-commitment), [Paper](https://www.sciencedirect.com/science/article/pii/S0378426624001080) | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)                  | Classify climate-related text into "climate commitments and actions" or not.                                                  | block         |
| ```climate_sentiment```             | Bingler et al.       | [Huggingface](https://huggingface.co/climatebert/distilroberta-base-climate-sentiment), [Paper](https://www.sciencedirect.com/science/article/pii/S0378426624001080)  | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)                  | Classify climate-related text into climate-related "sentiment classes", either opportunity, neutral, or risk.                 | block         |
| ```environmental_claim```           | Stammbach et al.     | [Huggingface](https://huggingface.co/climatebert/environmental-claims), [Paper](https://aclanthology.org/2023.acl-short.91/)                                          | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)                  | Classify text into environmental claim or not. The model is trained on the EnvironmentalClaims dataset.                       | sentence      |
| ```esg_bert```                      | Mukherjee et al.     | [Huggingface](https://huggingface.co/nbroad/ESG-BERT), [Blog](https://towardsdatascience.com/nlp-meets-sustainable-investing-d0542b3c264b)                            | [Apache 2.0 (Github)](https://github.com/mukut03/ESG-BERT?tab=Apache-2.0-1-ov-file#readme)                              | Classify text into 26 ESG-related topics. The full list of labels can be found [here]().                                      | sentence      |
| ```netzero_reduction```             | Schimanski et al.    | [Huggingface](https://huggingface.co/climatebert/netzero-reduction), [Paper](https://aclanthology.org/2023.emnlp-main.975/)                                           | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)                  | Classify climate-related text into net-zero target, reduction target, or no-target.                                           | block         |
| ```sst2```                          | DistilBERT community | [Huggingface](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english), [Related paper](https://www.mdpi.com/2076-3417/12/11/5614)          | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)                  | Classify text into positive or negative.                                                                                      | sentence      |
| ```transition_physical_renewable``` | Deng et al.          | [Huggingface](https://huggingface.co/climatebert/transition-physical), [Paper](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4080181)                           | [Apache 2.0](https://huggingface.co/datasets/choosealicense/licenses/blob/main/markdown/apache-2.0.md)                  | Classify text into transition risk exposure, physical risk exposure, or transition risk exposure related to renewable energy. | block         |


### Using custom huggingface models

Want to use some of your favorite huggingface models? 
You can use special annotator of ```custom_huggingface```.
For example, you can use [FinanceInc/auditor_sentiment_finetuned](https://huggingface.co/FinanceInc/auditor_sentiment_finetuned) as the annotator as follows.

```bash
python -m reportparse.main \
  -i reportparse/asset/example.pdf \
  -o ./results \
  --reader pymupdf \
  --annotators "custom_huggingface" \
  --custom_huggingface_annotator_name "auditor_sentiment" \
  --custom_huggingface_model_name_or_path "FinanceInc/auditor_sentiment_finetuned" \
  --custom_huggingface_level "block"
```


## FAQs

- I have faced errors when installing ReportParse.
  - We provide installation examples on Google Colab notebooks. Unless you face erros on the notebooks, the problem would be on your own environment. We do not consider any inquiries in this case.
- Can we use it for any PDF files other than sustainability reports?
  - Technically yes. However, we do not actively support genral reports or other PDF files.
- We want to extract more fine-grained document structure for my own report.
  - Unfortunately, we do not want to support _any_ documents. Reports are usually unstructured documents represented in a PDF file, and it is impossible to support all of them. Instead, we want to implement more general methods that can apply for various type of reports. 
- Is ReportParse reliable enough?
  - We plan to add some test codes to ensure the functional correctness. Please keep in mind that our tool may contain any bugs. Do not hesitate to point out these bugs if you find.
- Does ReportParse support tables and figures?
  - Currently, no.

## Future work

- LLM support
- Table support
- Keyword extraction
- Span-level annotation




