from dataclasses import dataclass
from typing import Dict, Any
import loguru
from abc import ABC, abstractmethod
from playwright.async_api import async_playwright
import asyncio
import re


@dataclass
class ScrapedContent:
    """Standardized scraped content format."""

    url: str
    html: str
    text: str
    status_code: int
    metadata: Dict[str, Any] = None


class Scraper(ABC):
    """Abstract base class for scrapers."""

    @abstractmethod
    async def setup(self):
        """Initialize the scraper resources."""
        pass

    @abstractmethod
    async def teardown(self):
        """Clean up the scraper resources."""
        pass

    @abstractmethod
    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Scrape a URL and return standardized content."""
        pass


class PlaywrightScraper:
    """Playwright-based scraper implementation."""

    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        self.headless = headless
        self.browser_type = browser_type
        self.browser = None
        self.context = None

    async def setup(self):
        """Initialize Playwright browser and context."""

        self.playwright = await async_playwright().start()

        browser_method = getattr(self.playwright, self.browser_type)
        self.browser = await browser_method.launch(headless=self.headless)
        self.context = await self.browser.new_context()

        loguru.logger.info(
            f"Playwright {self.browser_type} browser initialized in {'headless' if self.headless else 'headed'} mode"
        )

    async def teardown(self):
        """Clean up Playwright resources."""
        if self.browser:
            await self.browser.close()
        if hasattr(self, "playwright") and self.playwright:
            await self.playwright.stop()
        loguru.logger.info("Playwright resources cleaned up")

    async def scrape(self, url: str, **kwargs) -> ScrapedContent:
        """Scrape a URL using Playwright and return standardized content."""
        loguru.logger.info(f"{url} scraped")
        if not self.browser:
            await self.setup()

        try:
            page = await self.context.new_page()

            # Set default timeout
            timeout = kwargs.get("timeout", 30000)
            page.set_default_timeout(timeout)

            # Navigate to URL
            loguru.logger.info("Navigate to URL:"+url)
            response = await page.goto(url, wait_until="networkidle")
            status_code = response.status if response else 0

            # Get HTML and text content
            title = await page.title()
            html = await page.content()

            # ------- MOST IMPORTANT COMMENT IN THE REPO -------
            # Extract only user-visible text content from the page
            # This excludes: hidden elements, navigation dropdowns, collapsed accordions,
            # inactive tabs, script/style content, SVG code, HTML comments, and metadata
            # Essentially captures what a human would see when viewing the page
            text = await page.evaluate("document.body.innerText")

            # Close the page
            await page.close()

            return ScrapedContent(
                url=url,
                html=html,
                text=text,
                status_code=status_code,
                metadata={
                    "title": title,
                    "headers": response.headers if response else {},
                },
            )

        except Exception as e:
            loguru.logger.error(f"Error scraping {url}: {str(e)}")
            return ScrapedContent(
                url=url, html="", text="", status_code=0, metadata={"error": str(e)}
            )


def clean_scraped_text(text):
    """
    清理爬取的文本内容，移除HTML/CSS代码片段

    参数:
        text (str): 原始爬取的文本内容

    返回:
        str: 清理后的文本内容
    """
    # 移除CSS样式定义
    text = re.sub(r'[a-zA-Z#\.\-\_\s,:]+ \{[^\}]*\}', '', text)

    # 移除HTML标签
    text = re.sub(r'<[^>]*>', '', text)

    # 移除URL引用
    text = re.sub(r'url\([^\)]*\)', '', text)

    # 移除background-image等样式属性
    text = re.sub(r'background-[a-z\-]+:[^;]*;', '', text)

    # 移除width, height等样式属性
    text = re.sub(r'(width|height|display|float|vertical-align):[^;]*;', '', text)

    # 移除只包含空白字符的行
    text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)

    # 移除连续的多个空行，保留单个空行
    text = re.sub(r'\n\s*\n', '\n\n', text)

    return text.strip()


async def main():
    # 配置日志
    loguru.logger.add("scraper.log", rotation="10 MB")

    # 创建爬虫实例，headless=True表示无界面模式
    scraper = PlaywrightScraper(headless=True)

    try:
        # 要爬取的网址
        target_url = "https://blog.sciencenet.cn/blog-2089193-1469701.html"
        target_url = "https://blog.csdn.net/2401_85375151/article/details/144805338"
        target_url = "https://zhuanlan.zhihu.com/p/19647641182"
        # 爬取内容
        result = await scraper.scrape(target_url)

        # 清理文本内容
        cleaned_text = clean_scraped_text(result.text)

        # 打印爬取结果
        print(f"URL: {result.url}")
        print(f"状态码: {result.status_code}")
        print(f"标题: {result.metadata.get('title', 'N/A')}")
        print("\n-----清理后的文本内容预览(前500字)-----")
        print(cleaned_text[:500] + "...")

        # 保存完整HTML到文件
        with open("sciencenet_blog.html", "w", encoding="utf-8") as f:
            f.write(result.html)

        # 保存原始提取的文本到文件
        with open("sciencenet_blog_raw.txt", "w", encoding="utf-8") as f:
            f.write(result.text)

        # 保存清理后的文本到文件
        with open("sciencenet_blog_cleaned.txt", "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        print("\n爬取内容已保存到文件，包括原始HTML、原始文本和清理后的文本")

    except Exception as e:
        loguru.logger.error(f"爬虫运行出错: {str(e)}")
    finally:
        # 清理资源
        await scraper.teardown()

# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())