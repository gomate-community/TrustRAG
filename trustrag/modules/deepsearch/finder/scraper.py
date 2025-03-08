from dataclasses import dataclass
from typing import Dict, Any
# from trustrag.modules.deepsearch.utils import logger
import loguru
from abc import ABC, abstractmethod
from playwright.async_api import async_playwright

# ---- Data Models ----


@dataclass
class ScrapedContent:
    """Standardized scraped content format."""

    url: str
    html: str
    text: str
    status_code: int
    metadata: Dict[str, Any] = None


# ---- Scraper Interfaces ----


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