"""Model usage reporting API."""
from __future__ import annotations
from fastapi import APIRouter, Depends, HTTPException
from app.api.v1.auth import current_user
from app.schemas.auth import UserOut
from app.observability.core import trace_report, usage_dashboard, usage_summary
router=APIRouter(prefix="/api/v1/models",tags=["models"])
@router.get("/usage")
def model_usage(provider: str|None=None, model: str|None=None, operation: str|None=None, date_from: str|None=None, date_to: str|None=None, user: UserOut=Depends(current_user)):
    return {"items": usage_summary({"owner_id": str(user.id), "provider":provider,"model":model,"operation":operation,"date_from":date_from,"date_to":date_to})}


@router.get("/dashboard")
def model_dashboard(days: int = 14, user: UserOut = Depends(current_user)):
    return usage_dashboard(user.id, days=days)


@router.get("/traces/{trace_id}")
def get_trace(trace_id: str, user: UserOut = Depends(current_user)):
    value = trace_report(trace_id, owner_id=user.id)
    if not value:
        raise HTTPException(status_code=404, detail="Trace not found")
    return value
