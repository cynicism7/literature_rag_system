import re

YEAR_RE = re.compile(r"(19|20)\d{2}")

def extract_title(text: str) -> str:
    """
    Heuristic: first non-empty line
    """
    for line in text.splitlines():
        line = line.strip()
        if len(line) > 10:
            return line[:300]
    return ""


def extract_year(text: str) -> int | None:
    years = YEAR_RE.findall(text)
    if not years:
        return None
    # 取最早一个，通常是发表年
    return int("".join(years[0]))


def extract_abstract(text: str) -> str:
    """
    Try to extract abstract section
    """
    lower = text.lower()
    idx = lower.find("abstract")
    if idx == -1:
        return ""

    snippet = text[idx: idx + 3000]
    lines = snippet.splitlines()[1:10]
    abstract = " ".join(l.strip() for l in lines if len(l.strip()) > 5)
    return abstract[:2000]


def extract_metadata(text: str) -> dict:
    return {
        "title": extract_title(text),
        "year": extract_year(text),
        "abstract": extract_abstract(text)
    }
