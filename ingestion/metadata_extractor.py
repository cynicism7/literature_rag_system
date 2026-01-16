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


def extract_authors(text: str) -> str:
    """
    提取作者信息
    常见格式：
    1. "Authors: Name1, Name2, Name3"
    2. "Author: Name"
    3. "By: Name1, Name2"
    4. 标题后的第一行包含作者
    """
    text_lower = text.lower()

    # 方法1: 查找 "Authors:" 或 "Author:" 关键词
    patterns = [
        r"(?:authors?|by)[:\s]+([^\n]+)",
        r"received[:\s]+\d+[^\n]*\n[^\n]*\n[^\n]*\n([^\n]+)",  # Received后的几行可能是作者
    ]

    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            authors = match.group(1).strip()
            # 清理常见后缀
            authors = re.sub(r'\s*(received|accepted|published|revised).*$', '', authors, flags=re.IGNORECASE)
            authors = re.sub(r'\s*\d{4}.*$', '', authors)  # 移除年份
            if len(authors) > 3 and len(authors) < 500:  # 合理的作者长度
                return authors.strip()

    # 方法2: 查找标题后的几行，通常包含作者信息
    lines = text.splitlines()
    for i, line in enumerate(lines[:20]):  # 检查前20行
        line = line.strip()
        # 跳过标题行（通常较长）
        if len(line) > 10 and len(line) < 200:
            # 检查是否包含常见的作者格式（逗号分隔的姓名）
            if ',' in line and not line.lower().startswith(('abstract', 'keywords', 'introduction')):
                # 可能是作者行
                authors = re.sub(r'\s*\d{4}.*$', '', line)  # 移除年份
                authors = re.sub(r'\s*(received|accepted|published|revised).*$', '', authors, flags=re.IGNORECASE)
                if len(authors) > 3:
                    return authors.strip()

    return ""


def extract_year(text_blocks):
    text = "\n".join(text_blocks) if isinstance(text_blocks, list) else text_blocks

    matches = re.findall(r'\b(?:19|20)\d{2}\b', text)
    return matches[0] if matches else None


def extract_abstract(text_blocks):
    text = "\n".join(text_blocks) if isinstance(text_blocks, list) else text_blocks
    lower = text.lower()

    if "abstract" not in lower:
        return None

    start = lower.find("abstract")
    end = lower.find("keywords", start)

    if end == -1:
        end = start + 1500

    return text[start:end].strip()




def extract_metadata(text: str) -> dict:
    return {
        "title": extract_title(text),
        "authors": extract_authors(text),
        "year": extract_year(text),
        "abstract": extract_abstract(text)
    }
