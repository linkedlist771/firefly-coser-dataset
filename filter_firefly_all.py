from pathlib import Path
from configs import RESOURCES_DIR
from tqdm import tqdm
import cv2
from loguru import logger
import asyncio


CUTOFF_RESOLUTION = 500


def filter_watermark(source_dir: Path, target_dir: Path):
    
    # Detect

    # Inpaint

    raise NotImplementedError("Not implemented")

async def main():
    source_dir = RESOURCES_DIR / "firefly_images"
    target_dir = RESOURCES_DIR / "firefly_images_filtered"
    target_dir.mkdir(parents=True, exist_ok=True)
    files = list(source_dir.iterdir())
    success_count = 1
    for file in tqdm(files):
        if file.is_file():
            try:
                image = cv2.imread(str(file))
                if image.shape[0] < CUTOFF_RESOLUTION or image.shape[1] < CUTOFF_RESOLUTION:
                    pass
                else:
                    # all into the PNG 
                    cv2.imwrite(str(target_dir / f"{success_count}.png"), image)
                    success_count += 1
                    logger.info(f"Saved image {success_count} to {target_dir / f"{success_count:06d}.png"}")
            except Exception as e:
                logger.error(f"Error reading image {file}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
            




