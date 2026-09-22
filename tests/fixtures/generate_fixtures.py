"""PDF Fixture Generator for DocuChat unit tests.

Generated using `pypdf` (installed via requirements.txt) to produce:
1. `sample_2page.pdf`: A 2-page text PDF with known content on each page.
2. `empty_scanned.pdf`: A 2-page PDF with blank pages (no extractable text).
"""

from pathlib import Path
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject


def create_sample_2page_pdf(output_path: Path) -> None:
    """Create a 2-page PDF with distinct text streams using pypdf."""
    writer = PdfWriter()

    # Standard font dictionary for Type1 Helvetica
    font_dict = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    resources = DictionaryObject({
        NameObject("/Font"): DictionaryObject({
            NameObject("/F1"): font_dict
        })
    })

    # Page 1
    page1 = writer.add_blank_page(width=612, height=792)
    stream1 = DecodedStreamObject()
    stream1.set_data(
        b"BT /F1 12 Tf 72 712 Td (DocuChat Page 1: Retrieval Augmented Generation document ingestion test.) Tj ET"
    )
    page1[NameObject("/Contents")] = stream1
    page1[NameObject("/Resources")] = resources

    # Page 2
    page2 = writer.add_blank_page(width=612, height=792)
    stream2 = DecodedStreamObject()
    stream2.set_data(
        b"BT /F1 12 Tf 72 712 Td (DocuChat Page 2: Accurate citation metadata with 1-indexed page numbers.) Tj ET"
    )
    page2[NameObject("/Contents")] = stream2
    page2[NameObject("/Resources")] = resources

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)


def create_empty_scanned_pdf(output_path: Path) -> None:
    """Create a 2-page PDF without any text stream (simulating image/scanned PDF)."""
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    writer.add_blank_page(width=612, height=792)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    create_sample_2page_pdf(fixtures_dir / "sample_2page.pdf")
    create_empty_scanned_pdf(fixtures_dir / "empty_scanned.pdf")
    print(f"Fixtures generated successfully in {fixtures_dir}")
