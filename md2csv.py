from configs import MD_FILES, CSV_FILES
from utils import print_md, parse_md_link
import pandas as pd
import markdown

MD_FILE = MD_FILES / "link_source1.md"

content = MD_FILE.read_text()
content = parse_md_link(content)
html = markdown.markdown(content, extensions=["tables"])
df = pd.read_html(html)[0]
print(df)

CSV_FILE = CSV_FILES / "link_source1.csv"
df.to_csv(CSV_FILE, index=False)
