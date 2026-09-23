from scripts.prepare_benchmark_review import page_candidates


def test_chapter_two_footer_uses_printed_page_not_publication_number():
    text = (
        "Chapter 2\n"
        "2-1. A sufficiently long sentence about intelligence operations.\n"
        "01 October 2023 FM 2-0 2-1"
    )
    rows = page_candidates(text, pdf_page=45, chapter=2, doc_id="fm2-0-2023")
    assert len(rows) == 1
    assert rows[0]["page"] == "2-1"


def test_chapter_two_even_page_footer_order():
    text = (
        "Chapter 2\n"
        "2-2. A sufficiently long sentence about intelligence operations.\n"
        "2-2 FM 2-0 01 October 2023"
    )
    rows = page_candidates(text, pdf_page=46, chapter=2, doc_id="fm2-0-2023")
    assert len(rows) == 1
    assert rows[0]["page"] == "2-2"
