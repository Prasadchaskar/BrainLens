from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    predicted_class: str = Field(
        ...,
        description="Predicted brain tumor class.",
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Prediction confidence.",
    )

    model_name: str = Field(
        ...,
        description="Registered MLflow model name.",
    )

    model_version: str = Field(
        ...,
        description="Concrete registered MLflow model version.",
    )
    
    model_alias: str = Field(
        ...,
        description="MLflow alias used for inference.",
    )

    model_uri: str = Field(
        ...,
        description="MLflow model URI used by the API.",
    )