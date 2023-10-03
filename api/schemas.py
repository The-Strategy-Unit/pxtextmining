from pydantic import BaseModel, validator


class Test(BaseModel):
    test: str

    class Config:
        schema_extra = {"example": {"test": "Hello"}}


class ItemIn(BaseModel):
    comment_id: str
    comment_text: str
    question_type: str

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "question_type": "nonspecific",
            }
        }

    @validator("question_type")
    def question_type_validation(cls, v):
        if v not in ["what_good", "could_improve", "nonspecific"]:
            raise ValueError(
                "question_type must be one of what_good, could_improve, or nonspecific"
            )
        return v


class MultilabelOut(BaseModel):
    comment_id: str
    comment_text: str
    labels: list

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
                "labels": ["Staff manner & personal attributes", "Parking"],
            }
        }
