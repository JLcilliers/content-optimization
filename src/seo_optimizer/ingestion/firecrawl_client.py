"""
Firecrawl API Client - Fetch web page content via Firecrawl API.

This module provides HTTP client functionality to fetch page content
from URLs using the Firecrawl API, returning markdown content that
can be parsed into the DocumentAST format.
"""

import httpx
from dataclasses import dataclass, field
from typing import Any


class FirecrawlError(Exception):
    """Base exception for Firecrawl API errors."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class FirecrawlRateLimitError(FirecrawlError):
    """Raised when rate limited by Firecrawl API."""

    def __init__(self, message: str = "Too many requests, please wait"):
        super().__init__(message, status_code=429)


class FirecrawlBlockedError(FirecrawlError):
    """Raised when the site cannot be scraped."""

    def __init__(self, message: str = "This site cannot be analyzed"):
        super().__init__(message, status_code=403)


class FirecrawlTimeoutError(FirecrawlError):
    """Raised when the request times out."""

    def __init__(self, message: str = "Page took too long to load"):
        super().__init__(message, status_code=504)


@dataclass
class FirecrawlMetadata:
    """Metadata extracted from a scraped page."""

    title: str | None = None
    description: str | None = None
    language: str | None = None
    og_title: str | None = None
    og_description: str | None = None
    og_url: str | None = None
    og_image: str | None = None
    source_url: str | None = None
    status_code: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FirecrawlMetadata":
        """Create metadata from Firecrawl response dict."""
        return cls(
            title=data.get("title"),
            description=data.get("description"),
            language=data.get("language"),
            og_title=data.get("ogTitle"),
            og_description=data.get("ogDescription"),
            og_url=data.get("ogUrl"),
            og_image=data.get("ogImage"),
            source_url=data.get("sourceURL"),
            status_code=data.get("statusCode"),
        )


@dataclass
class FirecrawlResponse:
    """Response from Firecrawl scrape operation."""

    success: bool
    url: str
    markdown: str
    html: str | None = None
    metadata: FirecrawlMetadata = field(default_factory=FirecrawlMetadata)
    raw_html: str | None = None
    links: list[str] = field(default_factory=list)
    actions: dict[str, Any] | None = None
    warning: str | None = None

    @classmethod
    def from_api_response(cls, data: dict[str, Any], url: str) -> "FirecrawlResponse":
        """Create response from Firecrawl API response dict."""
        response_data = data.get("data", {})

        return cls(
            success=data.get("success", False),
            url=url,
            markdown=response_data.get("markdown", ""),
            html=response_data.get("html"),
            metadata=FirecrawlMetadata.from_dict(response_data.get("metadata", {})),
            raw_html=response_data.get("rawHtml"),
            links=response_data.get("links", []),
            actions=response_data.get("actions"),
            warning=response_data.get("warning"),
        )


class FirecrawlClient:
    """HTTP client for Firecrawl API."""

    def __init__(
        self,
        api_key: str,
        api_url: str = "https://api.firecrawl.dev/v1",
        timeout: int = 30,
    ):
        """
        Initialize Firecrawl client.

        Args:
            api_key: Firecrawl API key
            api_url: Base URL for Firecrawl API
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.api_url = api_url.rstrip("/")
        self.timeout = timeout

    def _get_headers(self) -> dict[str, str]:
        """Get request headers with authentication."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    async def scrape(
        self,
        url: str,
        formats: list[str] | None = None,
        only_main_content: bool = True,
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        wait_for: int = 0,
    ) -> FirecrawlResponse:
        """
        Scrape a URL and return its content.

        Args:
            url: The URL to scrape
            formats: Output formats to request (default: ["markdown"])
            only_main_content: Whether to extract only main content
            include_tags: HTML tags to include
            exclude_tags: HTML tags to exclude
            wait_for: Milliseconds to wait before scraping (for JS content)

        Returns:
            FirecrawlResponse with scraped content

        Raises:
            FirecrawlError: Base error for API issues
            FirecrawlRateLimitError: Rate limit exceeded
            FirecrawlBlockedError: Site cannot be scraped
            FirecrawlTimeoutError: Request timed out
        """
        if not self.api_key:
            raise FirecrawlError("Firecrawl API key not configured", status_code=500)

        # Build request payload
        payload: dict[str, Any] = {
            "url": url,
            "formats": formats or ["markdown"],
            "onlyMainContent": only_main_content,
        }

        if include_tags:
            payload["includeTags"] = include_tags
        if exclude_tags:
            payload["excludeTags"] = exclude_tags
        if wait_for > 0:
            payload["waitFor"] = wait_for

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    f"{self.api_url}/scrape",
                    headers=self._get_headers(),
                    json=payload,
                )

                # Handle error responses
                if response.status_code == 429:
                    raise FirecrawlRateLimitError()
                elif response.status_code == 403:
                    raise FirecrawlBlockedError()
                elif response.status_code == 408 or response.status_code == 504:
                    raise FirecrawlTimeoutError()
                elif response.status_code >= 400:
                    error_detail = "Unknown error"
                    try:
                        error_data = response.json()
                        error_detail = error_data.get("error", error_data.get("message", str(error_data)))
                    except Exception:
                        error_detail = response.text or f"HTTP {response.status_code}"
                    raise FirecrawlError(f"Firecrawl API error: {error_detail}", status_code=response.status_code)

                # Parse successful response
                data = response.json()
                return FirecrawlResponse.from_api_response(data, url)

        except httpx.TimeoutException:
            raise FirecrawlTimeoutError()
        except httpx.RequestError as e:
            raise FirecrawlError(f"Request failed: {str(e)}")

    def scrape_sync(
        self,
        url: str,
        formats: list[str] | None = None,
        only_main_content: bool = True,
        include_tags: list[str] | None = None,
        exclude_tags: list[str] | None = None,
        wait_for: int = 0,
    ) -> FirecrawlResponse:
        """
        Synchronous version of scrape for non-async contexts.

        Args:
            Same as scrape()

        Returns:
            FirecrawlResponse with scraped content
        """
        import asyncio

        return asyncio.run(
            self.scrape(
                url=url,
                formats=formats,
                only_main_content=only_main_content,
                include_tags=include_tags,
                exclude_tags=exclude_tags,
                wait_for=wait_for,
            )
        )
