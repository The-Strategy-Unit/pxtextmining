from pydantic import BaseModel


class Test(BaseModel):
    test: str

    class Config:
        schema_extra = {"example": {"test": "Hello"}}


class ItemIn(BaseModel):
    comment_id: str
    comment_text: str

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "comment_text": "Nurses were friendly. Parking was awful.",
            }
        }


class MultilabelOut(BaseModel):
    comment_id: str
    labels: list

    class Config:
        schema_extra = {
            "example": {
                "comment_id": "01",
                "labels": ["Staff manner & personal attributes", "Parking"],
            }
        }
