"""
scrape.py
Scrapes book listings from books.toscrape.com across multiple categories.

Dynamically discovers categories from the homepage sidebar (rather than
hardcoding category URLs, which include unpredictable numeric IDs) and
paginates through each category until all books are collected.

Stops once we have >= MIN_BOOKS books across >= MIN_CATEGORIES categories
(finishing the current category fully before stopping, so category counts
stay clean).

Output: raw_books.csv — one row per book, unprocessed fields as scraped.
"""

import time
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE_URL = "http://books.toscrape.com/"
MIN_BOOKS = 60
MIN_CATEGORIES = 3
REQUEST_DELAY_SECONDS = 0.5  # be polite to the practice site


def get_categories() -> list[tuple[str, str]]:
    """Return a list of (category_name, category_url) discovered from the homepage sidebar."""
    resp = requests.get(BASE_URL, timeout=10)
    resp.raise_for_status()
    resp.encoding = "utf-8"  # books.toscrape.com doesn't always declare charset correctly,
    # which makes requests misdetect it and mangle the £ symbol (e.g. "£53.74" -> "Â£53.74").
    # Forcing utf-8 here fixes that at the source.
    soup = BeautifulSoup(resp.text, "html.parser")

    categories = []
    for a in soup.select(".side_categories ul li ul li a"):
        name = a.get_text(strip=True)
        url = urljoin(BASE_URL, a["href"])
        categories.append((name, url))
    return categories


def scrape_category(name: str, start_url: str) -> list[dict]:
    """Scrape every book across all paginated pages of a single category."""
    books = []
    page_url = start_url

    while page_url:
        resp = requests.get(page_url, timeout=10)
        resp.raise_for_status()
        resp.encoding = "utf-8"  # see note in get_categories() above
        soup = BeautifulSoup(resp.text, "html.parser")

        for article in soup.select("article.product_pod"):
            title = article.h3.a["title"]
            price_text = article.select_one(".price_color").get_text(strip=True)

            star_tag = article.select_one("p.star-rating")
            star_classes = star_tag["class"]  # e.g. ["star-rating", "Three"]
            star_rating = next(c for c in star_classes if c != "star-rating")

            availability = article.select_one(".instock.availability").get_text(strip=True)

            books.append(
                {
                    "title": title,
                    "price": price_text,
                    "star_rating": star_rating,
                    "availability": availability,
                    "category": name,
                }
            )

        next_link = soup.select_one("li.next a")
        page_url = urljoin(page_url, next_link["href"]) if next_link else None
        if page_url:
            time.sleep(REQUEST_DELAY_SECONDS)

    return books


def main():
    categories = get_categories()
    if not categories:
        raise RuntimeError(
            "No categories found on the homepage — the site's HTML structure may "
            "have changed. Check the .side_categories selector in scrape_category()."
        )

    all_books: list[dict] = []
    used_categories: list[str] = []

    for name, url in categories:
        print(f"Scraping category: {name} ...")
        books = scrape_category(name, url)
        if not books:
            continue

        all_books.extend(books)
        used_categories.append(name)
        print(f"  -> {len(books)} books (running total: {len(all_books)})")

        if len(all_books) >= MIN_BOOKS and len(used_categories) >= MIN_CATEGORIES:
            break
        time.sleep(REQUEST_DELAY_SECONDS)

    df = pd.DataFrame(all_books)
    df.to_csv("raw_books.csv", index=False)

    print()
    print(f"Done. Scraped {len(df)} books across {df['category'].nunique()} categories:")
    print(df["category"].value_counts())
    print("Saved to raw_books.csv")


if __name__ == "__main__":
    main()
