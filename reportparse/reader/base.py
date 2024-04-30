import argparse
from typing import List
from registrable import Registrable

from reportparse.structure.document import Document


class BaseReader(Registrable):

    def __init__(self):
        return

    def read(
            self,
            input_path: str, max_pages: int = None, skip_pages: List[int] = None, skip_load_image: bool = False,
            args=None
    ) -> Document:
        raise NotImplementedError

    def add_argument(self, parser: argparse.ArgumentParser):
        raise NotImplementedError


