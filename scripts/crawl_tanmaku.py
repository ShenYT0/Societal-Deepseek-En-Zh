""""""
import os
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Sequence, Optional


def download_danmaku(bv: str) -> None:
    """Download danmaku from a single bv with BBDown"""
    os.system(f"bbdown --danmaku-only {bv}")
    
def download_subtitle(bv: str) -> None:
    """Download subti from a single bv with BBDown"""
    os.system(f"bbdown --sub-only --skip-ai=false {bv}")


def main(bv_list: list[str], save_dir: Path) -> None:
    """Download danmaku from a list of bv to save_dir"""
    # Logger config
    logging.basicConfig(level=logging.INFO)
    # Env variable setting
    os.chdir(save_dir)
    # Download
    for bv in bv_list:
        # download_danmaku(bv)
        download_subtitle(bv)


if __name__ == '__main__':
    with open("../data/raw/tech/bvs.txt") as f:
        bv_list = f.readlines()

    save_dir = Path("../data/clean/tech")

    main(bv_list=bv_list, save_dir=save_dir)
