import argparse
from registrable import Registrable

from reportparse.structure.document import Document


class BaseAnnotator(Registrable):

    def __init__(self):
        return

    def annotate(self, document: Document, args=None) -> Document:
        raise NotImplementedError

    def add_argument(self, parser: argparse.ArgumentParser):
        raise NotImplementedError


