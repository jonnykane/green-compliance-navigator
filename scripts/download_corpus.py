#!/usr/bin/env python3
"""
Download real regulatory PDFs for the UK Green Compliance Navigator corpus.

Run from the project root:
    python scripts/download_corpus.py

Downloads to: corpus/real/
Skips files that already exist (safe to re-run).
"""

import os
import time
import urllib.request
import urllib.error
from pathlib import Path

CORPUS_DIR = Path(__file__).parent.parent / "corpus" / "real"

DOCUMENTS = [
    {
        "filename": "tcfd_2017_recommendations.pdf",
        "url": "https://assets.bbhub.io/company/sites/60/2021/10/FINAL-2017-TCFD-Report.pdf",
        "description": "TCFD 2017 Recommendations Report",
        "publication_date": "2017-06-29",
        "issuing_body": "Task Force on Climate-related Financial Disclosures (TCFD)",
        "doc_type": "real",
    },
    {
        "filename": "secr_environmental_reporting_guidelines_2019.pdf",
        "url": "https://assets.publishing.service.gov.uk/media/67161e8696def6d27a4c9ab3/environmental-reporting-guidance-secr-march-2019.pdf",
        "description": "Environmental Reporting Guidelines: Including Streamlined Energy and Carbon Reporting (SECR) Guidance",
        "publication_date": "2019-03-29",
        "issuing_body": "Department for Business, Energy and Industrial Strategy (BEIS)",
        "doc_type": "real",
    },
    {
        "filename": "ppn_0621_carbon_reduction_plans.pdf",
        "url": "https://assets.publishing.service.gov.uk/media/62066d5ae90e077f7dec749e/PPN-0621-Taking-account-of-Carbon-Reduction-Plans-Jan22__1_.pdf",
        "description": "Procurement Policy Note 06/21: Taking Account of Carbon Reduction Plans in the Procurement of Major Government Contracts",
        "publication_date": "2022-01-01",
        "issuing_body": "Cabinet Office",
        "doc_type": "real",
    },
    {
        "filename": "ppn_0621_technical_standard_crp.pdf",
        "url": "https://assets.publishing.service.gov.uk/media/60ba4d208fa8f57ce980b5b7/PPN_0621_Technical_standard_for_the_Completion_of_Carbon_Reduction_Plans__2_.pdf",
        "description": "PPN 06/21 Technical Standard for the Completion of Carbon Reduction Plans",
        "publication_date": "2021-06-05",
        "issuing_body": "Cabinet Office",
        "doc_type": "real",
    },
]


def download_file(url: str, dest_path: Path, description: str) -> bool:
    """Download a single file with progress reporting. Returns True on success."""
    if dest_path.exists():
        size_kb = dest_path.stat().st_size / 1024
        print(f"  ✓ Already exists ({size_kb:.0f} KB): {dest_path.name}")
        return True

    print(f"  ↓ Downloading: {description}")
    print(f"    URL: {url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (compatible; GreenComplianceNavigator/1.0; "
            "research project)"
        )
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as response:
            content = response.read()

        dest_path.write_bytes(content)
        size_kb = len(content) / 1024
        print(f"    ✓ Saved ({size_kb:.0f} KB): {dest_path.name}")
        return True

    except urllib.error.HTTPError as e:
        print(f"    ✗ HTTP error {e.code}: {e.reason}")
        return False
    except urllib.error.URLError as e:
        print(f"    ✗ URL error: {e.reason}")
        return False
    except Exception as e:
        print(f"    ✗ Unexpected error: {e}")
        return False


def write_metadata(documents: list) -> None:
    """Write a simple metadata manifest for the real corpus documents."""
    import json

    manifest_path = CORPUS_DIR.parent / "metadata" / "real_corpus_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = []
    for doc in documents:
        manifest.append(
            {
                "filename": doc["filename"],
                "path": f"corpus/real/{doc['filename']}",
                "description": doc["description"],
                "publication_date": doc["publication_date"],
                "issuing_body": doc["issuing_body"],
                "doc_type": doc["doc_type"],
                "source_url": doc["url"],
            }
        )

    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\n  ✓ Manifest written: {manifest_path.name}")


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    print("UK Green Compliance Navigator — Corpus Download")
    print("=" * 55)
    print(f"Destination: {CORPUS_DIR}\n")

    results = []
    for i, doc in enumerate(DOCUMENTS, 1):
        print(f"[{i}/{len(DOCUMENTS)}] {doc['description'][:60]}...")
        dest = CORPUS_DIR / doc["filename"]
        success = download_file(doc["url"], dest, doc["description"])
        results.append((doc["filename"], success))
        if i < len(DOCUMENTS):
            time.sleep(1)  # polite delay between requests

    write_metadata(DOCUMENTS)

    print("\nSummary")
    print("-" * 40)
    success_count = sum(1 for _, ok in results if ok)
    for filename, ok in results:
        status = "✓" if ok else "✗"
        print(f"  {status} {filename}")

    print(f"\n{success_count}/{len(DOCUMENTS)} documents downloaded successfully.")

    if success_count < len(DOCUMENTS):
        print(
            "\nFor any failed downloads, you can download the files manually "
            "and place them in corpus/real/ with the filenames shown above."
        )


if __name__ == "__main__":
    main()
