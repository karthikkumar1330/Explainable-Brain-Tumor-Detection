from pydantic import BaseModel, Field
from typing import Optional


class PatientIntake(BaseModel):
    """Pydantic model to validate patient clinical demographics intake requests."""
    patient_id: str = Field(..., min_length=2, description="Unique Patient Identifier")
    name: str = Field(..., min_length=2, description="Full Legal Name")
    age: int = Field(..., ge=0, le=120, description="Age in Years")
    gender: str = Field("Female", description="Gender (e.g., Female, Male, Other)")
    ref_physician: str = Field("Dr. Unknown", description="Referring Physician Name")
    pixel_spacing_mm: float = Field(1.0, gt=0.0, description="MRI spatial scale pixel spacing in mm")
    xai_method: Optional[str] = Field("gradcam", description="Explainability method: gradcam, gradcam_plus_plus, or eigencam")
    ensemble_mode: Optional[bool] = Field(False, description="Whether to run multi-model ensemble prediction and calculate model agreement")



class SearchQuery(BaseModel):
    """Pydantic model to validate database history filter queries."""
    patient_id: Optional[str] = Field(None, description="Search Patient ID (supports wildcards)")
    scan_date: Optional[str] = Field(None, description="Search Acquisition Date (supports wildcards)")
