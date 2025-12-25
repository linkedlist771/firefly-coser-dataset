import re

from rich.console import Console
from rich.markdown import Markdown
from loguru import logger
from httpx import AsyncClient
from httpx_curl_cffi import AsyncCurlTransport
from curl_cffi.const import CurlOpt
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm.asyncio import tqdm as async_tqdm
from asyncio import TaskGroup
import asyncio
import pandas as pd
from configs import CSV_FILES, RESOURCES_DIR

def print_md(md: str):
    console = Console()
    console.print(Markdown(md))


def parse_md_link(md: str) -> str:
    """
    通过正则提取出链接，把 [text](link) 转换为 link
    """
    # 匹配 markdown 链接格式 [text](url)
    pattern = r"\[([^\]]*)\]\(([^)]+)\)"
    # 将所有 [text](link) 替换为 link
    result = re.sub(pattern, r"\2", md)
    return result


def create_curl_client(
    impersonate: str = "chrome",
    timeout: float = 600.0,
) -> AsyncClient:
    transport = AsyncCurlTransport(
        impersonate=impersonate,
        default_headers=False, 
        curl_options={
            CurlOpt.FRESH_CONNECT: True, 
        },
    )
    return AsyncClient(
        transport=transport,
        timeout=timeout,
        follow_redirects=True,
    )


# <img alt="" class="lazyload" data-src="https://avatar2.bahamut.com.tw/avataruserpic/l/e/leisure/leisure.png" id="avatar_fpath126785">

def _extract_image_url(img_tag: str) -> str:
    pattern = r'(?:src|data-src)=["\']?(https://[^"\'\s>]+)'
    match = re.search(pattern, img_tag)
    if match:
        return match.group(1)
    return ""

def parse_image_urls_from_html(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    image_urls = []
    for img in soup.find_all("img"):
        # logger.debug(f"img:\n{img}")
        # image_urls.append(img["src"])
        # we need a regex based 
        image_url = _extract_image_url(str(img))
        if image_url:
            image_urls.append(image_url)

    return image_urls

def filter_image_urls(image_urls: list[str]) -> list[str]:
    filtered_image_urls = []
    for image_url in image_urls:
        if image_url.endswith(".svg"):
            pass

        else:
            filtered_image_urls.append(image_url)
    return filtered_image_urls


async def scrape_image_urls(url: str) -> list[str]:
    try:
        client = create_curl_client()
        response = await client.get(url)
        image_urls = parse_image_urls_from_html(response.text)
        image_urls = filter_image_urls(image_urls)
        return image_urls
    except Exception as e:
        logger.error(f"Failed to scrape {url}: {e}")
        return []


async def download_image(url: str, target_path: Path) -> bool:
    try:
        parent_dir = target_path.parent
        parent_dir.mkdir(parents=True, exist_ok=True)
        client = create_curl_client()
        response = await client.get(url)
        with open(target_path, "wb") as f:
            f.write(response.content)
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def get_image_filename(url: str, index: int) -> str:
    url_path = url.split("?")[0]  # 去掉查询参数
    ext = Path(url_path).suffix.lower()
    if ext not in [".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"]:
        ext = ".jpg"  # 默认扩展名
    return f"{index:06d}{ext}"


# https://www.weibo.com/6305408483/QdnGWcoQv



async def scrape_single_image_url(url: str, target_dir: Path) -> bool:
    image_urls = await scrape_image_urls(url)
    if len(image_urls) == 0:
        return False
    target_dir.mkdir(parents=True, exist_ok=True)
    download_tasks = [download_image(image_url, target_dir / get_image_filename(image_url, idx)) for idx, image_url in enumerate(image_urls)]   
    async for result in async_tqdm(
        asyncio.as_completed(download_tasks),
        total=len(download_tasks),
        desc="Downloading images"
    ):
        if await result:
            success_count += 1
    logger.info(f"Downloaded {success_count}/{len(image_urls)} images to {target_dir}")
    return True


# if __name__ == "__main__":
#     asyncio.run(main())

if __name__ == "__main__":
    url = "https://www.91shenshi.com/posts/tomoyo-jiang-xing-qiong-tie-dao-liu-ying-qi-pao/"
    target_dir = RESOURCES_DIR / "images" / "tomoyo-jiang-xing-qiong-tie-dao-liu-ying-qi-pao"
    asyncio.run(scrape_single_image_url(url, target_dir))
