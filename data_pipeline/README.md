# Data Pipeline

Scrapes catalog-style pricing/availability data from books.toscrape.com, cleans
it, enriches it with a fixed-rate currency conversion, and loads it into a
normalized SQLite database — then queries it with both SQL and pandas.

## What this module does
- Scrapes all books across 3 categories (dynamically discovered from the site's
  own category sidebar, so it doesn't rely on hardcoded/fragile category IDs)
- Cleans and types the scraped fields: `price_gbp` (float), `rating` (int 1-5),
  `in_stock` (bool)
- Computes `price_inr` from the project's fixed baseline rate: **1 GBP = 105.50 INR**
- Loads into a two-table SQLite schema (`categories`, `books`) with a proper PK/FK relationship
- Runs 6 SQL queries collectively covering SELECT/WHERE, ORDER BY, LIMIT, DISTINCT,
  IN, BETWEEN, and a JOIN
- Reads 2 of those query results back via `pd.read_sql`, and separately reproduces
  the JOIN query using `pd.merge` on in-memory DataFrames — confirmed to produce
  identical output (see `queries.py` output)

## Files
- `scrape.py` — discovers categories, scrapes all books per category (with
  pagination), saves `raw_books.csv`
- `clean_and_load.py` — cleans/types the raw data, builds `zepto.db`
- `queries.py` — runs the 6 required SQL queries + the `pd.read_sql` vs `pd.merge` check

## Setup

```bash
cd data_pipeline
pip install -r requirements.txt
```

## How to run

```bash
python scrape.py            # -> raw_books.csv
python clean_and_load.py    # -> zepto.db
python queries.py           # prints all query output + the equivalence check
```

## Design decisions

**Category discovery:** rather than hardcoding category URLs (which include
unpredictable numeric IDs baked into the slug, e.g. `travel_2`), `scrape.py`
reads the homepage's own category sidebar to get category names and URLs.
This is more robust — it keeps working even if the site's internal IDs change.

**Currency conversion:** `price_inr = price_gbp * 105.50`, using the project's
fixed, keyless baseline rate stated in the assignment (no live/date-based
lookup, per the required graded path).

**Error handling during cleaning:**
- **Price parse failures** (unexpected text where a price was expected) are
  **median-imputed** rather than dropped — price is a continuous field, and a
  reasonable estimate keeps the row usable for downstream analysis rather than
  losing a whole book's record over one bad field.
- **Rating parse failures** (an unrecognized star-rating word) cause the row
  to be **dropped** rather than imputed — rating is a categorical/ordinal
  label, and fabricating a specific "median" rating for a specific book felt
  less honest than simply excluding a rare corrupted row.

**Schema:** two tables — `categories(category_id PK, category_name UNIQUE)`
and `books(book_id PK, title, price_gbp, price_inr, rating, in_stock,
category_id FK -> categories.category_id)`.

**Tie-breaking in the JOIN query:** the join query originally sorted only by
`rating DESC`, which meant `pd.read_sql` and `pd.merge` could return the same
10 books in different orders whenever multiple books tied on rating at the
LIMIT boundary. Fixed by adding `book_id ASC` as a secondary, deterministic
sort key in both the SQL query and the pandas `.sort_values()` call — this is
what makes the two outputs match exactly, not just contain the same rows.

