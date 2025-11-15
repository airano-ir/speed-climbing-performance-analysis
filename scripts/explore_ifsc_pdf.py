"""Explore IFSC PDF structure to understand route map layout."""

import pdfplumber
from pathlib import Path

def explore_pdf(pdf_path: str):
    """Explore PDF structure and extract text."""
    pdf_path = Path(pdf_path)

    print(f"Opening PDF: {pdf_path}")
    print("=" * 80)

    with pdfplumber.open(pdf_path) as pdf:
        print(f"Total pages: {len(pdf.pages)}\n")

        for i, page in enumerate(pdf.pages):
            print(f"\n{'=' * 80}")
            print(f"PAGE {i+1}")
            print('=' * 80)

            # Extract text
            text = page.extract_text()
            if text:
                lines = text.split('\n')
                print(f"Lines: {len(lines)}")
                print("\nFirst 30 lines:")
                print('-' * 80)
                for j, line in enumerate(lines[:30], 1):
                    print(f"{j:3d}: {line}")

                # Look for route-related keywords
                if any(keyword in text.lower() for keyword in ['route', 'hold', 'panel', 'grid', 'coordinate']):
                    print("\n⭐ This page contains route-related information!")

            # Extract tables
            tables = page.extract_tables()
            if tables:
                print(f"\n📊 Found {len(tables)} tables on this page")
                for t_idx, table in enumerate(tables):
                    print(f"\nTable {t_idx + 1}:")
                    print(f"Rows: {len(table)}, Columns: {len(table[0]) if table else 0}")
                    if table:
                        print("First 5 rows:")
                        for row in table[:5]:
                            print(row)

if __name__ == "__main__":
    pdf_path = "docs/IFSC_Speed_Licence_Rules.pdf"
    explore_pdf(pdf_path)
