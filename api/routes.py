#query接口
from fastapi import APIRouter
router = APIRouter()

@router.post("/query")
def query_endpoint(req):
    return {"status": "ok"}
