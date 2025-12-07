from bs4 import BeautifulSoup


def clean_summary(html: str) -> str:
    """ Convert HTML summary to plain text and drop 'Continue reading...' links """
    soup = BeautifulSoup(html, "html.parser")

    # Remove 'Continue reading...' anchors if present
    for a in soup.find_all("a"):
        if a.get_text(strip=True).lower().startswith("continue reading"):
            a.decompose()

    # Get plain text, with spaces between blocks
    return soup.get_text(" ", strip=True)