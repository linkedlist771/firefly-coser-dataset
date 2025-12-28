<div align="center">

# Firefly Coser Dataset

**An async web scraping toolkit for collecting cosplay image datasets**

[![Python](https://img.shields.io/badge/Python-3.13+-blue?logo=python&logoColor=white)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Project Structure](#project-structure) • [License](#license)

</div>

---

## Overview

Firefly Coser Dataset is a high-performance, asynchronous web scraping toolkit designed to collect cosplay images from various sources. Built with modern Python async/await patterns and featuring anti-detection capabilities through browser impersonation.

## Features

- **Async Architecture** — Leverages `asyncio` and `httpx` for concurrent downloads
- **Browser Impersonation** — Uses `curl_cffi` to bypass anti-bot protections
- **Smart Image Filtering** — Automatically filters out SVG icons and invalid images
- **Progress Tracking** — Real-time progress bars with `tqdm` for both scraping and downloading
- **Flexible Pipeline** — Convert Markdown tables to CSV, filter links, and batch download

## Installation

### Prerequisites

- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) (recommended) or pip

### Quick Start

```bash
# Clone the repository
git clone https://github.com/LLinkedlist/firefly-coser-dataset.git
cd firefly-coser-dataset

# Install dependencies with uv
uv sync

# Or with pip
pip install -e .
```

## Usage

### 1. Convert Markdown Links to CSV

```bash
python md2csv.py
```

This converts markdown tables with links in `mds/` to structured CSVs in `csvs/`.

### 2. Filter Links (Optional)

Open and run the Jupyter notebook for interactive filtering:

```bash
jupyter notebook notebooks/filter_csv_links.ipynb
```

### 3. Scrape and Download Images

```bash
python scrape_firefly_all.py
```

This will:
1. Read links from the CSV file
2. Asynchronously scrape all image URLs from each page
3. Deduplicate image URLs
4. Download all unique images to `resources/firefly_images/`

### Programmatic Usage

```python
from utils import scrape_image_urls, download_image, create_curl_client
import asyncio

async def example():
    # Scrape images from a URL
    image_urls = await scrape_image_urls("https://example.com/cosplay-gallery")
    
    # Download an image
    success = await download_image(
        url=image_urls[0],
        target_path=Path("./downloads/image.jpg")
    )

asyncio.run(example())
```

## Project Structure

```
firefly-coser-dataset/
├── configs.py              # Path configurations
├── utils.py                # Core utilities (scraping, downloading)
├── md2csv.py               # Markdown to CSV converter
├── scrape_firefly_all.py   # Main scraping script
├── mds/                    # Source markdown files
│   └── link_source1.md
├── csvs/                   # Generated CSV files
│   ├── link_source1.csv
│   └── link_source1_non_video_filtered.csv
├── notebooks/              # Jupyter notebooks for data processing
│   └── filter_csv_links.ipynb
└── resources/              # Downloaded images
    └── firefly_images/
```

## Tech Stack

| Category | Technology |
|----------|------------|
| HTTP Client | `httpx` + `httpx-curl-cffi` |
| Browser Impersonation | `curl_cffi` |
| HTML Parsing | `BeautifulSoup` + `lxml` |
| Data Processing | `pandas` |
| Progress Display | `tqdm` + `rich` |
| Logging | `loguru` |

## Configuration

All path configurations are centralized in `configs.py`:

```python
from configs import ROOT_DIR, RESOURCES_DIR, CSV_FILES, MD_FILES
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Made with ❤️ for the cosplay community**

</div>



