import re

from rich.console import Console
from rich.markdown import Markdown


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


# https://www.weibo.com/6305408483/QdnGWcoQv
