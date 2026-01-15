#请求/返回接口
from pydantic import BaseModel

class QueryRequest(BaseModel):
    query: str
    top_k: int = 50

class QueryResult(BaseModel):
    paper_id: str
    score: float
    reason: str
