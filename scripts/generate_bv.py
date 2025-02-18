""""""
from lxml import etree

import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional


def get_all_bv(html_content: str) -> list[str]:
    """Return all BV numbers from a Bilibili page"""
    tree = etree.HTML(html_content)
    elements = tree.xpath('//*[@data-key]')
    bvs = [element.get("data-key") for element in elements if element.get("data-key")]
    return bvs


def main():
    # Config
    html_file = "../data/raw/tech/videos_set.html"
    output_file = "../data/raw/tech/bvs.txt"

    # Get bvs
    with open(html_file, 'r') as f:
        html_content = f.read()
    bvs = get_all_bv(html_content)

    # Save to output
    with open(output_file, 'w') as f:
        f.writelines([bv + "\n" for bv in bvs])


if __name__ == '__main__':
    main()
