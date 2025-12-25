from pathlib import Path
from configs import RESOURCES_DIR




async def main():
    source_dir = RESOURCES_DIR / "firefly_images"
    target_dir = RESOURCES_DIR / "firefly_images_filtered"
    target_dir.mkdir(parents=True, exist_ok=True)





