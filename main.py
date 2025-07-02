from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime
import os

from api.schemas import (
    UserRegistrationRequest, UserRegistrationResponse,
    AuthenticationRequest, AuthenticationResponse,
    TrainingRequest, TrainingResponse,
    DatasetInfoResponse, UserListResponse,
    DeleteUserRequest, DeleteUserResponse,
    DeleteAllUsersResponse,
    HealthCheckResponse, ErrorResponse,
    UserDetailResponse, UserEmbeddingResponse, AllUsersDetailResponse
)
from services.fingerprint_service import FingerprintService

# Crear la aplicación FastAPI
app = FastAPI(
    title="Sistema de Autenticación de Huellas Dactilares",
    description="API REST para autenticación de huellas dactilares usando redes neuronales siamesas",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción, especificar los orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar el servicio
fingerprint_service = FingerprintService()

# Endpoints

@app.get("/", response_model=dict)
async def root():
    """Endpoint raíz con información básica"""
    return {
        "message": "Sistema de Autenticación de Huellas Dactilares",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Verificar el estado de salud del sistema"""
    try:
        health_status = fingerprint_service.get_health_status()
        return HealthCheckResponse(**health_status)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register", response_model=UserRegistrationResponse)
async def register_user(request: UserRegistrationRequest):
    """
    Registrar un nuevo usuario con sus huellas dactilares
    
    - **username**: Nombre del usuario
    - **images**: Lista de imágenes en formato base64 (máximo 10 imágenes)
    """
    try:
        result = fingerprint_service.register_user(request.username, request.images)
        return UserRegistrationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/authenticate", response_model=AuthenticationResponse)
async def authenticate_user(request: AuthenticationRequest):
    """
    Autenticar un usuario basado en su huella dactilar
    
    - **image**: Imagen en formato base64 para autenticación
    """
    try:
        result = fingerprint_service.authenticate_user(request.image)
        return AuthenticationResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingResponse)
async def train_model(request: TrainingRequest):
    """
    Entrenar el modelo siamesa
    
    - **dataset_path**: Ruta al archivo JSON del dataset
    - **epochs**: Número de épocas de entrenamiento (1-200)
    - **batch_size**: Tamaño del batch (8-128)
    - **validation_split**: Proporción de datos para validación (0.1-0.5)
    """
    try:
        result = fingerprint_service.train_model(
            dataset_path=request.dataset_path,
            epochs=request.epochs,
            batch_size=request.batch_size,
            validation_split=request.validation_split
        )
        return TrainingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dataset/info", response_model=DatasetInfoResponse)
async def get_dataset_info():
    """Obtener información del dataset"""
    try:
        dataset_info = fingerprint_service.get_dataset_info()
        return DatasetInfoResponse(**dataset_info)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users", response_model=UserListResponse)
async def get_all_users():
    """Obtener la lista de todos los usuarios registrados"""
    try:
        users = fingerprint_service.get_all_users()
        return UserListResponse(users=users, total_count=len(users))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/users/{username}", response_model=DeleteUserResponse)
async def delete_user(username: str):
    """
    Eliminar un usuario del sistema
    
    - **username**: Nombre del usuario a eliminar
    
    Este endpoint eliminará:
    - Los datos del usuario de la base de datos
    - Sus embeddings almacenados
    - Las imágenes asociadas
    """
    try:
        # Verificar que el usuario existe antes de intentar eliminarlo
        users = fingerprint_service.get_all_users()
        if username not in users:
            raise HTTPException(
                status_code=404, 
                detail=f"Usuario '{username}' no encontrado en el sistema"
            )
        
        result = fingerprint_service.delete_user(username)
        
        if result.get("success", False):
            return DeleteUserResponse(**result)
        else:
            raise HTTPException(
                status_code=500, 
                detail=result.get("message", "Error al eliminar el usuario")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/users", response_model=DeleteAllUsersResponse)
async def delete_all_users():
    """
    Eliminar todos los usuarios del sistema
    
    ⚠️ **CUIDADO**: Esta operación eliminará:
    - Todos los usuarios registrados
    - Todos los embeddings almacenados
    - Todas las imágenes asociadas
    
    Esta acción no se puede deshacer.
    """
    try:
        result = fingerprint_service.delete_all_users()
        
        if result.get("success", False):
            return DeleteAllUsersResponse(**result)
        else:
            raise HTTPException(
                status_code=500, 
                detail=result.get("message", "Error al eliminar todos los usuarios")
            )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/model/load")
async def load_model():
    """Cargar el modelo entrenado"""
    try:
        success = fingerprint_service.load_model()
        if success:
            return {"success": True, "message": "Modelo cargado exitosamente"}
        else:
            return {"success": False, "message": "No se encontró el modelo entrenado"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/detailed", response_model=AllUsersDetailResponse)
async def get_all_users_detailed():
    """Obtener información detallada de todos los usuarios incluyendo embeddings"""
    try:
        users_data = fingerprint_service.get_all_users_detailed()
        return AllUsersDetailResponse(**users_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{username}", response_model=UserDetailResponse)
async def get_user_details(username: str):
    """
    Obtener información detallada de un usuario específico
    
    - **username**: Nombre del usuario
    """
    try:
        result = fingerprint_service.get_user_details(username)
        if result.get("success", False):
            return UserDetailResponse(**result)
        else:
            raise HTTPException(status_code=404, detail=result.get("message", "Usuario no encontrado"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{username}/embeddings", response_model=UserEmbeddingResponse)
async def get_user_embeddings(username: str):
    """
    Obtener los embeddings de un usuario específico
    
    - **username**: Nombre del usuario
    """
    try:
        result = fingerprint_service.get_user_embeddings(username)
        if result.get("success", False):
            return UserEmbeddingResponse(**result)
        else:
            raise HTTPException(status_code=404, detail=result.get("message", "Usuario no encontrado"))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Manejo de errores global
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal Server Error",
            message=str(exc),
            timestamp=datetime.now()
        ).dict()
    )

# Manejo de errores 404
@app.exception_handler(404)
async def not_found_exception_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content=ErrorResponse(
            error="Not Found",
            message="El endpoint solicitado no existe",
            timestamp=datetime.now()
        ).dict()
    )

if __name__ == "__main__":
    # Crear directorios necesarios
    os.makedirs("models/saved", exist_ok=True)
    os.makedirs("data/dataset", exist_ok=True)
    os.makedirs("data/embeddings", exist_ok=True)
    
    # Ejecutar el servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
