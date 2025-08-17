from typing import List, Optional
from pydantic import BaseModel, Field

class TelecomRecord(BaseModel):
    trf: Optional[str] = None
    age: Optional[float] = None
    gndr: Optional[str] = None
    tenure: Optional[float] = None
    age_dev: Optional[float] = None
    dev_man: Optional[str] = None
    device_os_name: Optional[str] = None
    dev_num: Optional[float] = None
    is_dualsim: Optional[int] = Field(None, description="0/1")
    is_smartphone: Optional[int] = Field(None, description="0/1")
    is_featurephone: Optional[int] = None
    simcard_type: Optional[str] = None
    region: Optional[str] = None

class PredictRequest(BaseModel):
    records: List[TelecomRecord]

class PredictResponse(BaseModel):
    probabilities: List[float]
    labels: List[int]
