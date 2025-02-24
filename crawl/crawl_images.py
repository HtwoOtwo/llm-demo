import asyncio
import os

import requests
import urllib3
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

PROXIES = {
    'http': 'http://127.0.0.1:18080',
}

# browser config
proxy_config = {
    "server": "http://127.0.0.1:7890",
}
downloads_path = os.path.join(os.getcwd(), "my_downloads")
os.makedirs(downloads_path, exist_ok=True)
browser_config = BrowserConfig(proxy_config=proxy_config)

# RUN confg
crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        wait_for_images=True,
        scan_full_page=True,
    )

def download_image(url, filename, description=None):
    try:
        response = requests.get(url, stream=True, proxies=PROXIES, verify=False)
        response.raise_for_status()

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(1024):
                file.write(chunk)

        print(f"image saved: {filename}")

        if description:
            txt_path = os.path.splitext(filename)[0] + ".txt"
            with open(txt_path, 'wb') as file:
                file.write(description.encode('utf-8'))
            print(f"image description saved: {description}")
    except requests.exceptions.RequestException as e:
        print(f"error: {e}")

async def main():
    async with AsyncWebCrawler(config=browser_config) as crawler:
        result = await crawler.arun(
            url="https://www.nbcnews.com/business",
            config=crawl_config,
        )
        # print(result.markdown)
        if result.success:
            print("Media.Images count: ", len(result.media['images']))

            for image in result.media['images']:
                url = image['src']
                alt = image['alt']
                filename = os.path.basename(url)
                download_image(url, os.path.join(downloads_path, filename), description=alt)



if __name__ == "__main__":
    asyncio.run(main())
