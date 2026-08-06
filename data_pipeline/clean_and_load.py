"""
clean_and_load.py
Cleans raw_books.csv into properly typed columns, then loads the result into
a normalized two-table SQLite schema (categories + books, PK/FK related).

Run this AFTER scrape.py has produced raw_books.csv.

Design decisions (see README for the full write-up):
- price_gbp: strip the "£" symbol and parse as float. If parsing fails for a
  row (unexpected text), the price is left as NaN and later median-imputed —
  we impute rather than drop, since price is a continuous field where a
  reasonable estimate keeps the row usable for analysis.
- rating: mapped from the star-rating word ("One".."Five") to an integer 1-5.
  If a row has an unrecognized rating word (data corruption), the row is
  DROPPED rather than imputed — rating is a categorical/ordinal label, and
  guessing a "median" rating for a corrupted label risks fabricating a
  specific claim about a specific book, which feels less honest than simply
  excluding that row.
- in_stock: "In stock" (any casing/whitespace variant) -> True, else False.
- price_inr: computed as price_gbp * 105.50 — the project's fixed, keyless
  baseline rate (1 GBP = 105.50 INR), stated here and in the README exactly
  as required. No live/date-based lookup is used for the graded path.
"""

import sqlite3

import numpy as np
import pandas as pd

RAW_CSV = "raw_books.csv"
DB_PATH = "zepto.db"
GBP_TO_INR_RATE = 105.50  # fixed, project-defined baseline — see README

RATING_WORD_TO_INT = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def clean_price(price_text: str) -> float:
    """Strip currency symbol and parse to float. Returns NaN on failure."""
    try:
        return float(str(price_text).replace("£", "").strip())
    except (ValueError, TypeError):
        return np.nan


def clean_rating(star_word: str) -> float:
    """Map star-rating word to int 1-5. Returns NaN if the word isn't recognized."""
    return RATING_WORD_TO_INT.get(str(star_word).strip(), np.nan)


def clean_in_stock(availability_text: str) -> bool:
    """True if the availability text indicates the book is in stock."""
    return "in stock" in str(availability_text).strip().lower()


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(clean_rating)
    df["in_stock"] = df["availability"].apply(clean_in_stock)

    # Rating: drop rows with an unrecognized/missing rating word (see module docstring).
    unrecognized_rating = df["rating"].isna()
    if unrecognized_rating.any():
        print(f"Dropping {unrecognized_rating.sum()} row(s) with an unrecognized rating word.")
        df = df[~unrecognized_rating].copy()
    df["rating"] = df["rating"].astype(int)

    # Price: median-impute any rows that failed to parse.
    missing_price = df["price_gbp"].isna()
    if missing_price.any():
        median_price = df["price_gbp"].median()
        print(f"Median-imputing {missing_price.sum()} row(s) with unparseable price -> {median_price:.2f}")
        df["price_gbp"] = df["price_gbp"].fillna(median_price)

    df["price_inr"] = (df["price_gbp"] * GBP_TO_INR_RATE).round(2)

    return df[["title", "category", "price_gbp", "price_inr", "rating", "in_stock"]]


def build_database(df: pd.DataFrame, db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    cur.executescript(
        """
        DROP TABLE IF EXISTS books;
        DROP TABLE IF EXISTS categories;

        CREATE TABLE categories (
            category_id INTEGER PRIMARY KEY,
            category_name TEXT UNIQUE
        );

        CREATE TABLE books (
            book_id INTEGER PRIMARY KEY,
            title TEXT,
            price_gbp REAL,
            price_inr REAL,
            rating INTEGER,
            in_stock INTEGER,
            category_id INTEGER REFERENCES categories(category_id)
        );
        """
    )

    categories = sorted(df["category"].unique())
    category_to_id = {name: i + 1 for i, name in enumerate(categories)}
    cur.executemany(
        "INSERT INTO categories (category_id, category_name) VALUES (?, ?)",
        [(cid, name) for name, cid in category_to_id.items()],
    )

    books_rows = [
        (
            row.title,
            row.price_gbp,
            row.price_inr,
            int(row.rating),
            int(bool(row.in_stock)),
            category_to_id[row.category],
        )
        for row in df.itertuples()
    ]
    cur.executemany(
        """
        INSERT INTO books (title, price_gbp, price_inr, rating, in_stock, category_id)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        books_rows,
    )

    conn.commit()
    conn.close()


def main():
    raw = pd.read_csv(RAW_CSV)
    print(f"Loaded {len(raw)} raw rows from {RAW_CSV}")

    cleaned = clean_dataframe(raw)
    print(f"{len(cleaned)} rows after cleaning")

    build_database(cleaned)
    print(f"Loaded into {DB_PATH}: 'categories' and 'books' tables (PK/FK related).")


if __name__ == "__main__":
    main()
