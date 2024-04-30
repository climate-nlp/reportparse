import os
from reportparse.reader.base import BaseReader
from reportparse.annotator.base import BaseAnnotator

reader = BaseReader.by_name('pymupdf')()
document = reader.read(input_path='./reportparse/asset/example.pdf')

document = BaseAnnotator.by_name("environmental_claim")().annotate(document=document)
document = BaseAnnotator.by_name("sst2")().annotate(document=document)

os.makedirs('./results')

# Save the full data as a JSON file
document.save('./results/example.pdf.json')
# Save the easy-to-use dataset as a CSV file
document.to_dataframe(level='sentence').to_csv('./results/example.pdf.sentence-level-dataset.csv')
