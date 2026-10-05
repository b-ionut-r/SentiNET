"""util.gist: an 8-K excerpt as a short, reader-friendly line."""
from __future__ import annotations

import pytest

from app.analytics.util import gist


@pytest.mark.parametrize(("text", "expected"), [
    # Live GPRO: date preamble, defined term and state-of-incorporation boilerplate dropped.
    ("On September 2, 2026, GoPro, Inc., a Delaware corporation (the “Company”), entered into an Agreement and "
     "Plan of Merger with Action Acquisitions LLC. The merger is expected to close in Q4.",
     "GoPro, Inc. entered into an Agreement and Plan of Merger with Action Acquisitions LLC"),
    ("NVIDIA Corporation agreed to acquire Hugging Face, Inc. The price is $11.9 billion.",
     "NVIDIA Corporation agreed to acquire Hugging Face, Inc."),
    ("GoPro, Inc., a Delaware corporation, issued a convertible debenture of $50 million. More.",
     "GoPro, Inc. issued a convertible debenture of $50 million"),
    ("As previously disclosed, on May 4, 2026, Acme Ltd., a Cayman Islands exempted company, sold shares.",
     "Acme Ltd. sold shares"),
    ("GoPro, Inc. merged with Starman Optical, Inc., a Delaware corporation and a wholly owned subsidiary of "
     "Parent (“Merger Sub”). Next.", "GoPro, Inc. merged with Starman Optical, Inc."),
    # Abbreviations do not end the sentence; a sentence after one still does.
    ("The Company appointed Dr. Jane Roe as CFO. She replaces Mr. Smith.", "The Company appointed Dr. Jane Roe as CFO"),
    ("Approved the sale to U.S. Steel. Done.", "Approved the sale to U.S. Steel"),
    ("Merged with Parent Corp. and Merger Sub Inc. The merger closed.", "Merged with Parent Corp. and Merger Sub Inc."),
    ("Paid $3.5 million in cash. Next.", "Paid $3.5 million in cash"),
    # Honorifics and firm suffixes inside a sentence (item 5.02 departures, placement agents).
    ("On October 1, 2026, the Board appointed Mr. John Smith as Chief Executive Officer. He succeeds X.",
     "The Board appointed Mr. John Smith as Chief Executive Officer"),
    ("Effective October 1, 2026, Dr. Jane Doe resigned as Chief Financial Officer.",
     "Dr. Jane Doe resigned as Chief Financial Officer"),
    ("The Company entered into a placement agency agreement with H.C. Wainwright & Co. LLC. The offering closed.",
     "The Company entered into a placement agency agreement with H.C. Wainwright & Co. LLC"),
    ("The Company issued 1,000,000 shares to XYZ Co. Ltd. in a private placement.",
     "The Company issued 1,000,000 shares to XYZ Co. Ltd. in a private placement"),
    ("Acme Inc. announced St. Louis plant closure.", "Acme Inc. announced St. Louis plant closure"),
    # Other date preambles.
    ("Effective as of October 1, 2026, the Company completed the sale of its U.S. business to Acme Holdings, L.P.",
     "The Company completed the sale of its U.S. business to Acme Holdings, L.P."),
    ("On 4 May 2026, Acme plc entered into a deal.", "Acme plc entered into a deal"),
    # Ordinary appositives and short text are left alone.
    ("X spun off the unit, a new company, to holders.", "X spun off the unit, a new company, to holders"),
    ("Short", "Short"),
    ("", ""),
])
def test_gist(text: str, expected: str) -> None:
    assert gist(text) == expected


def test_gist_trims_long_text_on_a_word_boundary() -> None:
    out = gist("The Company entered into " + "a very long agreement " * 20, limit=60)
    assert len(out) <= 60 and out.endswith("…")


def test_gist_can_keep_every_sentence() -> None:
    text = ("On May 4, 2026, NVIDIA Corporation (the “Company”) agreed to acquire Hugging Face, Inc. The price "
            "is $11.9 billion.")
    assert gist(text, 200, first_sentence=False) == ("NVIDIA Corporation agreed to acquire Hugging Face, Inc. "
                                                     "The price is $11.9 billion")
