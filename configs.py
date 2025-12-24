from pathlib import Path

ROOT_DIR = Path(__file__).parent


RESOURCES_DIR = ROOT_DIR / "resources"

RESOURCES_DIR.mkdir(parents=True, exist_ok=True)

MD_FILES = ROOT_DIR / "mds"


MD_FILES.mkdir(parents=True, exist_ok=True)

CSV_FILES = ROOT_DIR / "csvs"

CSV_FILES.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    from loguru import logger

    logger.info(f"ROOT_DIR: {ROOT_DIR}")
