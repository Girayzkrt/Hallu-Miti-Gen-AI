import requests
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin

base_url = "https://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/oa_noncomm/xml/"
download_dir = "pmc_downloads"
os.makedirs(download_dir, exist_ok=True)

response = requests.get(base_url)
soup = BeautifulSoup(response.text, "html.parser")

for link in soup.find_all("a"):
    href = link.get("href")
    if href.endswith(".tar.gz"):
        file_url = urljoin(base_url, href)
        filename = os.path.join(download_dir, href)
        print(f"Downloading {href}...")
        with requests.get(file_url, stream=True) as r:
            with open(filename, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
print("Done.")
