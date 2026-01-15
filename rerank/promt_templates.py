#rerank prompt
RERANK_PROMPT = """
You are ranking academic papers.

Query:
{query}

Paper title:
{title}

Abstract:
{abstract}

Return JSON:
{{
  "relevance_score": 0-5,
  "keep": true or false,
  "reason": "short explanation"
}}
"""
