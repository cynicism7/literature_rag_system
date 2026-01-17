import re
import fitz  # PyMuPDF

YEAR_RE = re.compile(r"(19|20)\d{2}")

def extract_title(text: str) -> str:
    """
    提取论文标题
    策略：
    1. 标题通常在文档前部，在Abstract之前
    2. 标题通常较长（15-300字符），不包含特定的元数据关键词
    3. 由于文本可能没有换行（被替换为空格），需要基于位置和模式匹配
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    text_lower = text.lower()
    text_stripped = text.strip()
    
    # 找到Abstract的位置作为边界
    abstract_pos = text_lower.find('abstract')
    if abstract_pos > 0:
        search_text = text_stripped[:abstract_pos]
    else:
        # 如果没有Abstract，搜索前2000个字符
        search_text = text_stripped[:2000] if len(text_stripped) > 2000 else text_stripped
    
    # 要跳过的关键词（通常不是标题）
    skip_keywords = ['journal', 'volume', 'issue', 'pp.', 'doi:', 'received', 
                     'accepted', 'published', 'copyright', '©', 'issn', 
                     'article', 'proceedings', 'conference', 'workshop',
                     'authors:', 'author:', 'by:']
    
    # 尝试通过标点符号或关键词分割文本（因为可能没有换行）
    # 方法1: 尝试找到第一个较长的文本段（可能是标题）
    # 使用多个空格、句号、或特定关键词作为分割点
    parts = re.split(r'\s{3,}|\.\s+(?=[A-Z])|abstract|keywords|introduction', search_text, flags=re.IGNORECASE)
    
    best_title = ""
    best_score = 0
    
    for i, part in enumerate(parts[:20]):  # 只检查前20个部分
        part = part.strip()
        if not part:
            continue
        
        # 跳过太短或太长的部分
        if len(part) < 10 or len(part) > 400:
            continue
        
        part_lower = part.lower()
        
        # 跳过包含元数据关键词的部分
        if any(keyword in part_lower for keyword in skip_keywords):
            continue
        
        # 跳过看起来像作者的部分（包含多个逗号分隔的名字）
        if part.count(',') >= 3 or (',' in part and re.search(r'\b(university|institute|department|college)\b', part_lower)):
            continue
        
        # 跳过看起来像日期的部分
        if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', part_lower):
            continue
        
        # 跳过纯数字或特殊字符
        if re.match(r'^[\d\s\W]+$', part):
            continue
        
        # 标题评分
        score = 0
        
        # 长度评分：15-250字符比较好
        if 15 <= len(part) <= 250:
            score += 4
        elif 10 <= len(part) < 15 or 250 < len(part) <= 350:
            score += 2
        elif 350 < len(part) <= 400:
            score += 1
        
        # 位置评分：越靠前越好
        if i < 3:
            score += 3
        elif i < 6:
            score += 2
        elif i < 10:
            score += 1
        
        # 避免以特定标点结尾（除了问号、感叹号）
        if not part.rstrip().endswith((',', ';', ':')) or part.rstrip().endswith(('?', '!')):
            score += 1
        
        # 如果包含较多大写单词开头（可能是标题格式）
        words = part.split()
        if len(words) >= 3:
            cap_words = sum(1 for w in words[:15] if w and (w[0].isupper() or w.isupper()))
            if len(words) > 0 and cap_words / min(len(words), 15) > 0.4:
                score += 2
        
        # 如果包含常见标题词汇（如"of", "the", "and"等），可能是标题
        common_title_words = ['of', 'the', 'and', 'in', 'for', 'on', 'with', 'to', 'a', 'an']
        if any(word in part_lower for word in common_title_words):
            score += 1
        
        if score > best_score:
            best_score = score
            best_title = part[:300]  # 限制长度
    
    if best_title and best_score >= 3:  # 至少要有一定分数
        return best_title
    
    # 方法2: 如果上述方法失败，尝试提取前500个字符中第一个较长的文本段
    if abstract_pos > 0:
        first_part = text_stripped[:min(500, abstract_pos)].strip()
    else:
        first_part = text_stripped[:500].strip()
    
    # 移除开头的元数据关键词
    for keyword in skip_keywords:
        if first_part.lower().startswith(keyword):
            # 找到关键词后的内容
            idx = first_part.lower().find(keyword)
            if idx >= 0:
                first_part = first_part[idx + len(keyword):].strip()
                # 移除可能的分隔符
                first_part = re.sub(r'^[:\s]+', '', first_part)
    
    # 如果第一部分足够长且不包含太多逗号（可能是作者），返回它
    if len(first_part) >= 10 and len(first_part) <= 300 and first_part.count(',') < 3:
        # 尝试在第一个句号或特定关键词处截断
        for delimiter in ['. ', ' Abstract', ' Keywords', ' Introduction']:
            if delimiter.lower() in first_part.lower():
                idx = first_part.lower().find(delimiter.lower())
                if idx > 10:
                    first_part = first_part[:idx].strip()
                    break
        
        if len(first_part) >= 10:
            return first_part[:300]
    
    return ""


def extract_authors(text: str) -> str:
    """
    提取作者信息
    常见格式：
    1. "Authors: Name1, Name2, Name3"
    2. "Author: Name"
    3. "By: Name1, Name2"
    4. 标题后的文本包含作者（通常在Abstract之前）
    注意：由于文本可能没有换行，需要基于模式匹配
    """
    if not text or len(text.strip()) == 0:
        return ""
    
    text_lower = text.lower()
    text_stripped = text.strip()
    
    # 方法1: 精确匹配 "Authors:"、"Author:"、"By:" 关键词（优先级最高）
    # 使用更宽松的正则表达式
    author_label_patterns = [
        r'(?:^|\s)(?:authors?|by)\s*[:\s]+\s*([^\n]{10,300}?)(?:\s+(?:received|accepted|published|abstract|keywords|introduction|\d{4})|$)',  # Authors: ... 或 By: ...
        r'(?:^|\s)(?:corresponding\s+author|first\s+author)\s*[:\s]+\s*([^\n]{10,300}?)(?:\s+(?:received|accepted|published|abstract|keywords|\d{4})|$)',  # Corresponding author: ...
    ]
    
    for pattern in author_label_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            authors = match.group(1).strip()
            # 清理常见后缀和后置内容
            authors = re.sub(r'\s*(received|accepted|published|revised|submitted).*$', '', authors, flags=re.IGNORECASE)
            authors = re.sub(r'\s*\d{4}.*$', '', authors)  # 移除年份及之后的内容
            authors = re.sub(r'\s*doi[:\s].*$', '', authors, flags=re.IGNORECASE)  # 移除DOI
            
            # 验证：作者行应该包含姓名（通常包含逗号分隔或多个名字）
            if len(authors) >= 3 and len(authors) < 500:
                # 包含姓名特征：逗号分隔或"and"连接或姓名模式
                if ',' in authors or ' and ' in authors.lower() or re.search(r'\b[A-Z][a-z]+\s+[A-Z]', authors):
                    return authors.strip()

    # 方法2: 在标题和Abstract之间查找作者
    # 找到Abstract的位置
    abstract_pos = text_lower.find('abstract')
    
    # 先尝试找到标题的结束位置（标题通常在300-500字符内）
    # 找到第一个可能较长且不包含太多逗号的文本段作为标题
    title_end = 300  # 默认假设标题在前300字符内
    if abstract_pos > 0 and abstract_pos < 1000:
        # 如果Abstract在1000字符内，在标题和Abstract之间查找
        search_start = min(300, abstract_pos // 2)  # 从标题结束位置开始搜索
        search_text = text_stripped[search_start:abstract_pos] if abstract_pos > search_start else text_stripped[:abstract_pos]
    else:
        # 如果没有Abstract或Abstract太远，在300-1500字符之间查找
        search_text = text_stripped[300:1500] if len(text_stripped) > 1500 else text_stripped[300:] if len(text_stripped) > 300 else ""
    
    if not search_text or len(search_text.strip()) < 10:
        # 如果上述方法失败，搜索前2000字符
        search_text = text_stripped[:2000] if len(text_stripped) > 2000 else text_stripped
    
    # 尝试通过多个空格、句号、或特定关键词分割文本
    parts = re.split(r'\s{3,}|\.\s+(?=[A-Z])|abstract|keywords|introduction', search_text, flags=re.IGNORECASE)
    
    for part in parts[:20]:  # 检查前20个部分
        part = part.strip()
        if not part:
            continue
        
        # 跳过太短或太长的部分
        if len(part) < 5 or len(part) > 600:
            continue
        
        part_lower = part.lower()
        
        # 跳过明显的非作者行
        skip_patterns = ['abstract', 'keywords', 'introduction', 'journal', 'volume', 'issue', 'doi:']
        if any(pattern in part_lower for pattern in skip_patterns):
            continue
        
        # 跳过包含日期的部分（但允许年份）
        if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', part_lower):
            continue
        
        # 作者行的特征：
        # 1. 包含逗号分隔的名字（至少一个逗号）
        # 2. 包含"and"连接多个名字
        # 3. 包含名字模式（大写字母开头的单词，如 "John Smith"）
        has_comma = ',' in part
        has_and = ' and ' in part_lower
        has_name_pattern = re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+', part) or re.search(r'\b[A-Z]\.\s*[A-Z][a-z]+', part)
        
        # 如果满足作者特征
        if has_comma or has_and or has_name_pattern:
            authors = part
            
            # 如果包含机构关键词（university, institute等），尝试分割，只取作者部分
            institution_keywords = ['university', 'institute', 'college', 'department', 
                                   'laboratory', 'lab', 'school', 'center', 'centre']
            for keyword in institution_keywords:
                if keyword in part_lower:
                    # 找到关键词的位置
                    pos = part_lower.find(keyword)
                    if pos > 10:  # 确保关键词不在开头
                        # 取关键词前的内容
                        before_keyword = part[:pos].rstrip()
                        if before_keyword:
                            # 如果关键词前有逗号，尝试找到最后一个逗号前的内容
                            last_comma = before_keyword.rfind(',')
                            if last_comma > 0:
                                authors = before_keyword[:last_comma].rstrip()
                            else:
                                authors = before_keyword.rstrip()
                            break
            
            # 清理常见后缀
            authors = re.sub(r'\s*(received|accepted|published|revised|submitted).*$', '', authors, flags=re.IGNORECASE)
            authors = re.sub(r'\s*\d{4}.*$', '', authors)  # 移除年份及之后
            authors = re.sub(r'\s*doi[:\s].*$', '', authors, flags=re.IGNORECASE)  # 移除DOI
            authors = re.sub(r'\s*\([^)]*\)$', '', authors)  # 移除末尾括号
            
            # 验证长度和格式
            if len(authors) >= 3 and len(authors) < 500:
                # 确保包含姓名特征
                if ',' in authors or ' and ' in authors.lower() or re.search(r'\b[A-Z][a-z]+\s+[A-Z]', authors):
                    return authors.strip()

    # 方法3: 如果没有找到，尝试在前2000字符中直接查找包含多个逗号分隔的姓名模式
    search_text = text_stripped[:2000]
    # 查找包含逗号分隔名字的模式（更宽松的匹配）
    # 匹配类似 "Name1, Name2, Name3" 的模式
    comma_patterns = [
        r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.)?(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+[A-Z]\.)?)+)',  # 多个逗号分隔的完整名字
        r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+\s+[A-Z][a-z]+)+)',  # 简单格式：FirstName LastName, FirstName LastName
    ]
    
    for pattern in comma_patterns:
        matches = re.finditer(pattern, search_text)
        for match in matches:
            authors = match.group(1).strip()
            # 清理
            authors = re.sub(r'\s*(received|accepted|published|revised|submitted).*$', '', authors, flags=re.IGNORECASE)
            authors = re.sub(r'\s*\d{4}.*$', '', authors)
            # 如果找到机构关键词，截断
            for keyword in ['university', 'institute', 'college', 'department', 'laboratory', 'lab']:
                if keyword in authors.lower():
                    idx = authors.lower().find(keyword)
                    if idx > 10:
                        authors = authors[:idx].rstrip().rstrip(',')
                        break
            if len(authors) >= 5 and len(authors) < 500:
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




def extract_metadata_from_pdf(pdf_path: str) -> dict:
    """
    从PDF提取元数据，优先使用PDF内置元数据，然后使用布局分析，最后使用文本分析
    """
    title = None
    authors = None
    full_text = ""  # 用于year和abstract提取
    
    # 方法1: 尝试从PDF元数据提取（最可靠）
    try:
        doc = fitz.open(pdf_path)
        meta = doc.metadata
        
        # 验证并提取标题
        if meta.get('title') and _is_valid_title(meta['title']):
            title = meta['title'].strip()
        
        # 验证并提取作者
        if meta.get('author') and _is_valid_author(meta['author']):
            authors = meta['author'].strip()
        
        # 提取完整文本用于year和abstract
        full_text = "\n".join([page.get_text() for page in doc])
        
        # 如果元数据不完整，尝试从第一页布局提取
        if not title or not authors:
            first_page = doc[0]
            first_page_text = first_page.get_text()
            
            # 方法2: 使用布局分析（字体大小、位置）
            if not title:
                layout_title = _extract_title_from_layout(first_page, first_page_text)
                if layout_title:
                    title = layout_title
            if not authors:
                layout_authors = _extract_authors_from_layout(first_page, first_page_text)
                if layout_authors:
                    authors = layout_authors
        
        doc.close()
    except Exception as e:
        # 如果PDF读取失败，回退到文本分析
        try:
            doc = fitz.open(pdf_path)
            full_text = "\n".join([page.get_text() for page in doc])
            doc.close()
        except:
            pass
    
    # 方法3: 如果仍然失败，使用文本分析（原有方法）
    if full_text:
        if not title:
            title = extract_title(full_text)
        if not authors:
            authors = extract_authors(full_text)
    
    return {
        "title": title or "",
        "authors": authors or "",
        "year": extract_year(full_text) if full_text else None,
        "abstract": extract_abstract(full_text) if full_text else None
    }


def extract_metadata(text: str, pdf_path: str = None) -> dict:
    """
    提取元数据
    如果提供了pdf_path，优先使用PDF元数据和布局分析
    否则只使用文本分析
    """
    if pdf_path:
        return extract_metadata_from_pdf(pdf_path)
    else:
        return {
            "title": extract_title(text),
            "authors": extract_authors(text),
            "year": extract_year(text),
            "abstract": extract_abstract(text)
        }


def _is_valid_title(title: str) -> bool:
    """验证标题是否有效"""
    if not title or len(title.strip()) < 5:
        return False
    title_lower = title.lower().strip()
    # 排除占位符和无效标题
    invalid = ['untitled', 'title', 'document', 'new document', 'pdf', '']
    return title_lower not in invalid and not title_lower.startswith('untitled')


def _is_valid_author(author: str) -> bool:
    """验证作者是否有效"""
    if not author or len(author.strip()) < 3:
        return False
    author_lower = author.lower().strip()
    # 排除占位符
    invalid = ['author', 'anonymous', 'unknown', '']
    return author_lower not in invalid and any(c.isalpha() for c in author)


def _extract_title_from_layout(page, text: str) -> str:
    """
    从第一页布局提取标题
    策略：找到最大字体的文本块，位于页面顶部，在Abstract之前
    """
    try:
        # 获取所有文本块及其属性
        blocks = page.get_text("dict")
        
        # 找到Abstract的位置
        abstract_pos = text.lower().find('abstract')
        search_text = text[:abstract_pos] if abstract_pos > 0 else text[:2000]
        
        # 收集文本块，按字体大小和位置排序
        text_blocks = []
        for block in blocks.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line.get("spans", []):
                    text_blocks.append({
                        "text": span.get("text", "").strip(),
                        "size": span.get("size", 0),
                        "y0": line.get("bbox", [0, 0, 0, 0])[1],  # 上边界
                        "font": span.get("font", "")
                    })
        
        # 过滤和排序：排除太小的字体，按字体大小和位置排序
        text_blocks = [b for b in text_blocks if len(b["text"]) >= 10 and b["size"] > 8]
        if not text_blocks:
            return ""
        
        # 按字体大小降序排序，然后按位置（y坐标）升序排序
        text_blocks.sort(key=lambda x: (-x["size"], x["y0"]))
        
        # 检查前几个候选
        skip_keywords = ['journal', 'volume', 'issue', 'doi:', 'received', 
                        'accepted', 'published', 'copyright', '©', 'issn',
                        'authors:', 'author:', 'by:']
        
        for block in text_blocks[:10]:
            text = block["text"]
            if len(text) < 10 or len(text) > 400:
                continue
            
            text_lower = text.lower()
            # 跳过包含元数据关键词的块
            if any(kw in text_lower for kw in skip_keywords):
                continue
            
            # 跳过看起来像作者的行（多个逗号）
            if text.count(',') >= 3:
                continue
            
            # 如果文本在搜索范围内，返回它
            if text in search_text or search_text.startswith(text[:50]):
                return text[:300]
        
        # 如果没有找到，返回最大字体的文本（如果合理）
        if text_blocks:
            candidate = text_blocks[0]["text"]
            if 10 <= len(candidate) <= 400 and candidate.count(',') < 3:
                return candidate[:300]
    
    except Exception:
        pass
    
    return ""


def _extract_authors_from_layout(page, text: str) -> str:
    """
    从第一页布局提取作者 - 使用类似标题识别的高成功率方法
    策略：
    1. 找到标题的位置和字体大小
    2. 在标题下方查找字体稍小但大于正文的文本块
    3. 这些文本块通常包含作者信息（逗号分隔的名字）
    4. 在Abstract之前停止搜索
    """
    try:
        blocks = page.get_text("dict")
        page_height = page.rect.height
        
        # 找到Abstract的位置
        abstract_pos = text.lower().find('abstract')
        abstract_y = None
        
        # 收集所有文本行，包含位置和字体信息
        lines = []
        for block in blocks.get("blocks", []):
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text = ""
                line_sizes = []
                line_y0 = line.get("bbox", [0, 0, 0, 0])[1]
                line_y1 = line.get("bbox", [0, 0, 0, 0])[3]
                
                for span in line.get("spans", []):
                    span_text = span.get("text", "").strip()
                    if span_text:
                        line_text += span_text + " "
                        line_sizes.append(span.get("size", 0))
                
                if line_text.strip():
                    avg_size = sum(line_sizes) / len(line_sizes) if line_sizes else 0
                    lines.append({
                        "text": line_text.strip(),
                        "size": avg_size,
                        "y0": line_y0,
                        "y1": line_y1,
                        "y_center": (line_y0 + line_y1) / 2
                    })
                    
                    # 记录Abstract的位置
                    if abstract_y is None and 'abstract' in line_text.lower():
                        abstract_y = line_y0
        
        if not lines:
            return ""
        
        # 按y坐标排序（从上到下）
        lines.sort(key=lambda x: x["y0"])
        
        # 找到标题：通常是最大字体的文本块（在页面顶部）
        # 排除页眉（前15%的页面高度）
        header_threshold = page_height * 0.15
        content_lines = [l for l in lines if l["y0"] > header_threshold]
        
        if not content_lines:
            content_lines = lines
        
        # 找到最大字体（可能是标题）
        max_size = max(l["size"] for l in content_lines[:20])  # 只看前20行
        title_size = max_size
        
        # 标题通常在最大字体的90%以上
        title_threshold = max_size * 0.85
        title_lines = [l for l in content_lines if l["size"] >= title_threshold]
        
        if not title_lines:
            # 如果没有找到明显的标题，假设第一行是标题
            title_lines = [content_lines[0]] if content_lines else []
        
        # 找到标题结束位置
        title_end_y = max(l["y1"] for l in title_lines) if title_lines else header_threshold
        
        # 作者通常在标题下方，字体稍小但大于正文
        # 正文字体通常是页面中最常见的较小字体
        body_candidates = [l["size"] for l in content_lines if l["y0"] > page_height * 0.3 and l["size"] < title_size * 0.8]
        if body_candidates:
            from collections import Counter
            body_size = Counter(body_candidates).most_common(1)[0][0]
        else:
            body_size = title_size * 0.6  # 默认假设正文是标题的60%
        
        # 作者字体应该在标题和正文之间
        author_min_size = body_size * 1.1  # 比正文稍大
        author_max_size = title_size * 0.95  # 比标题稍小
        
        # 搜索范围：标题下方到Abstract之前（或页面前40%）
        search_end_y = abstract_y if abstract_y else page_height * 0.4
        
        # 收集作者候选行
        author_lines = []
        for line in content_lines:
            # 必须在标题下方
            if line["y0"] <= title_end_y:
                continue
            
            # 必须在搜索结束位置之前
            if line["y0"] >= search_end_y:
                break
            
            # 字体大小必须在作者范围内
            if not (author_min_size <= line["size"] <= author_max_size):
                continue
            
            line_text = line["text"]
            line_lower = line_text.lower()
            
            # 跳过明显的非作者行
            skip_patterns = ['abstract', 'keywords', 'introduction', 'journal', 'volume', 'issue', 'doi:']
            if any(p in line_lower for p in skip_patterns):
                continue
            
            # 跳过包含日期的行
            if re.search(r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\b', line_lower):
                continue
            
            # 检查作者特征：包含逗号、"and"或姓名模式
            has_comma = ',' in line_text
            has_and = ' and ' in line_lower
            has_name_pattern = re.search(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+', line_text) or re.search(r'\b[A-Z]\.\s*[A-Z][a-z]+', line_text)
            
            if has_comma or has_and or has_name_pattern:
                author_lines.append(line)
        
        # 如果找到作者行，合并它们（作者可能跨多行）
        if author_lines:
            # 按位置排序
            author_lines.sort(key=lambda x: x["y0"])
            
            # 合并相邻的作者行（如果它们很接近）
            merged_authors = []
            current_group = [author_lines[0]]
            
            for i in range(1, len(author_lines)):
                prev_line = author_lines[i-1]
                curr_line = author_lines[i]
                # 如果两行很接近（垂直距离小于字体大小的2倍），合并
                if curr_line["y0"] - prev_line["y1"] < prev_line["size"] * 2:
                    current_group.append(curr_line)
                else:
                    merged_authors.append(current_group)
                    current_group = [curr_line]
            
            if current_group:
                merged_authors.append(current_group)
            
            # 取第一组作者行（最靠近标题的）
            if merged_authors:
                author_group = merged_authors[0]
                author_text = " ".join([l["text"] for l in author_group])
                
                # 清理机构信息
                authors = author_text
                institution_keywords = ['university', 'institute', 'college', 'department', 
                                      'laboratory', 'lab', 'school', 'center', 'centre', 
                                      'email', '@', 'correspondence']
                
                for keyword in institution_keywords:
                    if keyword in authors.lower():
                        pos = authors.lower().find(keyword)
                        if pos > 10:
                            authors = authors[:pos].rstrip().rstrip(',')
                            break
                
                # 清理后缀
                authors = re.sub(r'\s*(received|accepted|published|revised|submitted).*$', '', authors, flags=re.IGNORECASE)
                authors = re.sub(r'\s*\d{4}.*$', '', authors)
                authors = re.sub(r'\s*doi[:\s].*$', '', authors, flags=re.IGNORECASE)
                authors = re.sub(r'\s*\([^)]*\)$', '', authors)  # 移除末尾括号
                authors = re.sub(r'\s*\[[^\]]*\]$', '', authors)  # 移除末尾方括号
                
                # 验证长度和格式
                if len(authors) >= 3 and len(authors) < 500:
                    # 确保包含姓名特征
                    if ',' in authors or ' and ' in authors.lower() or re.search(r'\b[A-Z][a-z]+\s+[A-Z]', authors):
                        return authors.strip()
    
    except Exception as e:
        # 如果布局分析失败，返回空字符串，让文本分析接管
        pass
    
    return ""
