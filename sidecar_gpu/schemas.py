from typing import Annotated, Literal
from pydantic import BaseModel, ConfigDict, Field


class MatrixJob(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["matrix.multiply"] = "matrix.multiply"
    a: str = Field(pattern=r"^o_[0-9a-f]{32}$")
    b: str = Field(pattern=r"^o_[0-9a-f]{32}$")
    a_shape: list[int] = Field(min_length=2, max_length=2)
    b_shape: list[int] = Field(min_length=2, max_length=2)
    dtype: Literal["float32"] = "float32"
    retain_output: bool = True
    retry_of: str | None = Field(default=None, pattern=r"^j_[0-9a-f]{32}$")


class EmbeddingJob(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    operation: Literal["image.embed"]
    images: list[Annotated[str, Field(pattern=r"^o_[0-9a-f]{32}$")]] = Field(min_length=1, max_length=16)
    model: Literal["mobilenet_v3_small.imagenet1k_v1.features576"] = "mobilenet_v3_small.imagenet1k_v1.features576"
    retain_output: bool = True
    retry_of: str | None = Field(default=None, pattern=r"^j_[0-9a-f]{32}$")
