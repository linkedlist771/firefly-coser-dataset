from configs import CSV_FILES, RESOURCES_DIR
from utils import scrape_image_urls, download_image, get_image_filename
import pandas as pd
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
from loguru import logger


async def main():
    csv_file = CSV_FILES / "link_source1.csv"
    df = pd.read_csv(csv_file)

    links = df["链接"].tolist()

    # Step 1: 爬取所有图片 URL
    logger.info("Step 1: Scraping image URLs...")
    tasks = [scrape_image_urls(link) for link in links]

    all_image_urls = []
    async for result in async_tqdm(
        asyncio.as_completed(tasks), total=len(tasks), desc="Scraping image URLs"
    ):
        image_urls = await result
        all_image_urls.extend(image_urls)

    logger.info(f"Total scraped image URLs: {len(all_image_urls)}")

    # 去重
    all_image_urls = list(set(all_image_urls))
    logger.info(f"Unique image URLs: {len(all_image_urls)}")

    # Step 2: 下载所有图片
    logger.info("Step 2: Downloading images...")
    download_dir = RESOURCES_DIR / "firefly_images"
    download_dir.mkdir(parents=True, exist_ok=True)

    download_tasks = []
    for idx, url in enumerate(all_image_urls):
        filename = get_image_filename(url, idx)
        target_path = download_dir / filename
        download_tasks.append(download_image(url, target_path))

    success_count = 0
    async for result in async_tqdm(
        asyncio.as_completed(download_tasks),
        total=len(download_tasks),
        desc="Downloading images",
    ):
        if await result:
            success_count += 1

    logger.info(
        f"Downloaded {success_count}/{len(all_image_urls)} images to {download_dir}"
    )


if __name__ == "__main__":
    asyncio.run(main())
