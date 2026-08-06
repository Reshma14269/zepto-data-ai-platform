"""
queries.py
Runs >= 5 SQL queries against zepto.db, collectively covering:
SELECT/WHERE, ORDER BY, LIMIT, DISTINCT, and (IN or BETWEEN) — plus >= 1 JOIN.

Then reads back >= 2 query results via pd.read_sql, and separately reproduces
the join query's result using pd.merge on in-memory DataFrames (no SQL),
printing both side by side to confirm they match.
"""

import sqlite3

import pandas as pd

DB_PATH = "zepto.db"

QUERIES = {
    "Q1_select_where": """
        SELECT title, price_gbp
        FROM books
        WHERE in_stock = 1
        LIMIT 10;
    """,
    "Q2_order_by_limit": """
        SELECT title, price_gbp
        FROM books
        ORDER BY price_gbp DESC
        LIMIT 10;
    """,
    "Q3_distinct": """
        SELECT DISTINCT category_id
        FROM books;
    """,
    "Q4_where_between": """
        SELECT title, price_gbp
        FROM books
        WHERE price_gbp BETWEEN 10 AND 20;
    """,
    "Q5_where_in": """
        SELECT title, rating
        FROM books
        WHERE rating IN (4, 5)
        LIMIT 10;
    """,
    "Q6_join": """
        SELECT b.book_id, b.title, c.category_name, b.rating
        FROM books b
        JOIN categories c ON b.category_id = c.category_id
        ORDER BY b.rating DESC, b.book_id ASC
        LIMIT 10;
    """,
}


def run_all_queries(conn: sqlite3.Connection):
    """Execute each query, printing the SQL and its output."""
    for name, sql in QUERIES.items():
        print(f"--- {name} ---")
        print(sql.strip())
        cur = conn.execute(sql)
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        print(pd.DataFrame(rows, columns=cols))
        print()


def verify_read_sql_vs_merge(conn: sqlite3.Connection):
    """
    Read back Q2 and Q6 via pd.read_sql (satisfies the ">= 2 queries via
    pd.read_sql" requirement), then separately reproduce Q6 (the join query)
    using pd.merge on in-memory DataFrames — no SQL involved in this second
    path — and show both outputs side by side to confirm they match.
    """
    print("=== Reading back via pd.read_sql ===")
    q2_df = pd.read_sql(QUERIES["Q2_order_by_limit"], conn)
    print("Q2 via pd.read_sql:")
    print(q2_df)
    print()

    q6_sql_df = pd.read_sql(QUERIES["Q6_join"], conn)
    print("Q6 (join) via pd.read_sql:")
    print(q6_sql_df)
    print()

    print("=== Reproducing the join with pd.merge (no SQL) ===")
    books_df = pd.read_sql("SELECT * FROM books;", conn)
    categories_df = pd.read_sql("SELECT * FROM categories;", conn)

    merged = books_df.merge(categories_df, on="category_id", how="inner")
    q6_merge_df = (
        merged[["book_id", "title", "category_name", "rating"]]
        .sort_values(["rating", "book_id"], ascending=[False, True])
        .head(10)
        .reset_index(drop=True)
    )
    print("Q6 (join) via pd.merge:")
    print(q6_merge_df)
    print()

    sql_sorted = q6_sql_df.reset_index(drop=True)
    merge_sorted = q6_merge_df.reset_index(drop=True)
    matches = sql_sorted.equals(merge_sorted)
    print(f"pd.read_sql result matches pd.merge result: {matches}")


def main():
    conn = sqlite3.connect(DB_PATH)
    run_all_queries(conn)
    verify_read_sql_vs_merge(conn)
    conn.close()


if __name__ == "__main__":
    main()
