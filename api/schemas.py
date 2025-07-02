from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class UserRegistrationRequest(BaseModel):
    """Esquema para el registro de usuarios"""
    username: str = Field(..., description="Nombre del usuario")
    images: List[str] = Field(..., min_items=1, max_items=10, description="Lista de imágenes en formato base64")

class UserRegistrationResponse(BaseModel):
    """Respuesta del registro de usuarios"""
    success: bool
    message: str
    username: Optional[str] = None
    embedding_count: Optional[int] = None

class AuthenticationRequest(BaseModel):
    """Esquema para la autenticación de usuarios"""
    image: str = Field(..., description="Imagen en formato base64 para autenticación")

class AuthenticationResponse(BaseModel):
    """Respuesta de la autenticación"""
    success: bool
    authenticated: bool
    username: Optional[str] = None
    similarity_score: Optional[float] = None
    message: str

class TrainingRequest(BaseModel):
    """Esquema para el entrenamiento del modelo"""
    dataset_path: str = Field(..., description="Ruta al archivo JSON del dataset")
    epochs: int = Field(default=50, ge=1, le=200, description="Número de épocas de entrenamiento")
    batch_size: int = Field(default=32, ge=8, le=128, description="Tamaño del batch")
    validation_split: float = Field(default=0.2, ge=0.1, le=0.5, description="Proporción de datos para validación")

class TrainingResponse(BaseModel):
    """Respuesta del entrenamiento"""
    success: bool
    message: str
    training_history: Optional[dict] = None
    final_accuracy: Optional[float] = None
    final_loss: Optional[float] = None

class UserInfo(BaseModel):
    """Información de un usuario"""
    username: str
    embedding_count: int
    registered_date: str
    image_paths: List[str]

class UserDetailResponse(BaseModel):
    """Información detallada de un usuario incluyendo embeddings"""
    username: str
    embedding_count: int
    registered_date: str
    image_paths: List[str]
    embeddings: Optional[List[List[float]]] = Field(None, description="Lista de embeddings del usuario")
    embedding_stats: Optional[dict] = Field(None, description="Estadísticas de los embeddings")

class UserEmbeddingResponse(BaseModel):
    """Respuesta con embeddings de un usuario específico"""
    success: bool
    username: str
    embedding_count: int
    embeddings: List[List[float]]
    embedding_dimension: int
    statistics: dict

class DatasetInfoResponse(BaseModel):
    """Información del dataset"""
    total_users: int
    total_embeddings: int
    users: List[str]
    dataset_path: str
    embeddings_path: str

class UserListResponse(BaseModel):
    """Lista de usuarios"""
    users: List[str]
    total_count: int

class DeleteUserRequest(BaseModel):
    """Esquema para eliminar usuario"""
    username: str = Field(..., description="Nombre del usuario a eliminar")

class DeleteUserResponse(BaseModel):
    """Respuesta de eliminación de usuario"""
    success: bool
    message: str

class HealthCheckResponse(BaseModel):
    """Respuesta del health check"""
    status: str
    timestamp: datetime
    model_loaded: bool
    dataset_info: Optional[DatasetInfoResponse] = None

class ErrorResponse(BaseModel):
    """Respuesta de error"""
    error: str
    message: str
    timestamp: datetime

class AllUsersDetailResponse(BaseModel):
    """Lista detallada de todos los usuarios"""
    users: List[UserDetailResponse]
    total_count: int
    total_embeddings: int

class DeleteAllUsersResponse(BaseModel):
    """Respuesta de eliminación de todos los usuarios"""
    success: bool
    message: str
    deleted_count: int
    deleted_users: List[str]