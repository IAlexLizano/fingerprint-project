"""
Servicio de Autenticación de Huellas Dactilares
Integrado con el modelo siamesa mejorado
"""
import os
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime
import cv2
import base64
import tensorflow as tf

from models.fingerprint_siamese_model import ImprovedSiameseNetwork
from data.fingerprint_data_processor import FingerprintDataProcessor
from data.dataset_manager import DatasetManager

class FingerprintService:
    def __init__(self, model_path: str = "best_fingerprint_model.h5"):
        """
        Inicializa el servicio de autenticación de huellas dactilares
        
        Args:
            model_path: Ruta al mejor modelo entrenado
        """
        self.model_path = model_path
        self.siamese_network = ImprovedSiameseNetwork()
        self.data_processor = FingerprintDataProcessor()
        self.dataset_manager = DatasetManager()
        
        # Cargar modelo si existe
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                # Cargar el modelo usando el método de la clase
                self.siamese_network = ImprovedSiameseNetwork.load_model(model_path)
                self.model_loaded = True
                print(f"✅ Modelo cargado exitosamente desde: {model_path}")
            except Exception as e:
                print(f"❌ Error al cargar el modelo: {e}")
                self.model_loaded = False
        else:
            print(f"⚠️ Modelo no encontrado en {model_path}")
            print("💡 Ejecuta 'python train_model.py' para entrenar el modelo")
            self.model_loaded = False
    
    def _generate_embedding_from_base64(self, image_base64: str) -> np.ndarray:
        """
        Genera embedding de una imagen en base64
        
        Args:
            image_base64: Imagen codificada en base64
            
        Returns:
            Array numpy con el embedding
        """
        if not self.model_loaded:
            raise Exception("Modelo no cargado. No se puede generar embedding.")
        
        try:
            # Decodificar imagen base64
            image_data = base64.b64decode(image_base64)
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise Exception("No se pudo decodificar la imagen")
            
            # Preprocesar imagen usando el procesador de datos
            # Redimensionar primero
            image = cv2.resize(image, self.data_processor.target_size, interpolation=cv2.INTER_AREA)
            
            # Aplicar CLAHE
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            image = clahe.apply(image)
            
            # Filtro gaussiano
            image = cv2.GaussianBlur(image, (3, 3), 0)
            
            # Realzar bordes
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            img_enhanced = cv2.filter2D(image, -1, kernel)
            image = cv2.addWeighted(image, 0.7, img_enhanced, 0.3, 0)
            image = np.clip(image, 0, 255)
            
            # Normalizar
            image = image.astype(np.float32) / 255.0
            
            # Añadir dimensiones de batch y canal
            processed_image = np.expand_dims(image, axis=[0, -1])
            
            # Generar embedding
            embedding = self.siamese_network.get_embeddings(processed_image)
            return embedding.flatten()
            
        except Exception as e:
            raise Exception(f"Error al generar embedding: {str(e)}")
    
    def register_user(self, username: str, images_base64: List[str]) -> Dict:
        """
        Registra un nuevo usuario con sus huellas dactilares
        
        Args:
            username: Nombre del usuario
            images_base64: Lista de imágenes codificadas en base64
            
        Returns:
            Diccionario con el resultado del registro
        """
        try:
            # Validaciones
            if not self.model_loaded:
                return {
                    "success": False,
                    "message": "Modelo no cargado. No se puede registrar usuario.",
                    "username": None,
                    "embedding_count": None
                }
            
            if not username or not username.strip():
                return {
                    "success": False,
                    "message": "El nombre de usuario no puede estar vacío.",
                    "username": None,
                    "embedding_count": None
                }
            
            if not images_base64 or len(images_base64) == 0:
                return {
                    "success": False,
                    "message": "Se requiere al menos una imagen para el registro.",
                    "username": None,
                    "embedding_count": None
                }
            
            # Verificar si el usuario ya existe
            if username in self.dataset_manager.get_all_users():
                return {
                    "success": False,
                    "message": f"El usuario '{username}' ya existe en el sistema.",
                    "username": None,
                    "embedding_count": None
                }
            
            # Generar embeddings
            embeddings = []
            for i, image_base64 in enumerate(images_base64):
                try:
                    embedding = self._generate_embedding_from_base64(image_base64)
                    embeddings.append(embedding)
                except Exception as e:
                    return {
                        "success": False,
                        "message": f"Error al procesar la imagen {i+1}: {str(e)}",
                        "username": None,
                        "embedding_count": None
                    }
            
            # Registrar usuario en el dataset
            success = self.dataset_manager.register_user(username, images_base64, embeddings)
            
            if success:
                return {
                    "success": True,
                    "message": f"Usuario '{username}' registrado exitosamente con {len(embeddings)} embeddings.",
                    "username": username,
                    "embedding_count": len(embeddings)
                }
            else:
                return {
                    "success": False,
                    "message": f"Error al registrar el usuario '{username}'.",
                    "username": None,
                    "embedding_count": None
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error interno del servidor: {str(e)}",
                "username": None,
                "embedding_count": None
            }
    
    def authenticate_user(self, image_base64: str, threshold: float = 0.75) -> Dict:
        """
        Autentica un usuario usando una huella dactilar
        
        Args:
            image_base64: Imagen de huella en base64
            threshold: Umbral de similitud para autenticación
            
        Returns:
            Diccionario con el resultado de la autenticación
        """
        try:
            # Validaciones
            if not self.model_loaded:
                return {
                    "success": False,
                    "authenticated": False,
                    "username": None,
                    "similarity_score": None,
                    "message": "Modelo no cargado. No se puede autenticar."
                }
            
            if not image_base64:
                return {
                    "success": False,
                    "authenticated": False,
                    "username": None,
                    "similarity_score": None,
                    "message": "Se requiere una imagen para la autenticación."
                }
            
            # Verificar que hay usuarios registrados
            if self.dataset_manager.get_user_count() == 0:
                return {
                    "success": False,
                    "authenticated": False,
                    "username": None,
                    "similarity_score": None,
                    "message": "No hay usuarios registrados en el sistema."
                }
            
            # Generar embedding de la imagen de consulta
            try:
                query_embedding = self._generate_embedding_from_base64(image_base64)
            except Exception as e:
                return {
                    "success": False,
                    "authenticated": False,
                    "username": None,
                    "similarity_score": None,
                    "message": f"Error al procesar la imagen: {str(e)}"
                }
            
            # Buscar el mejor match
            best_match, similarity_score = self.dataset_manager.find_best_match(
                query_embedding, threshold
            )
            
            if best_match is not None:
                return {
                    "success": True,
                    "authenticated": True,
                    "username": best_match,
                    "similarity_score": float(similarity_score),
                    "message": f"Usuario autenticado: {best_match}"
                }
            else:
                return {
                    "success": True,
                    "authenticated": False,
                    "username": None,
                    "similarity_score": float(similarity_score) if similarity_score is not None else 0.0,
                    "message": "Usuario no reconocido."
                }
                
        except Exception as e:
            return {
                "success": False,
                "authenticated": False,
                "username": None,
                "similarity_score": None,
                "message": f"Error interno del servidor: {str(e)}"
            }
    
    def get_user_count(self) -> int:
        """Obtiene el número de usuarios registrados"""
        return self.dataset_manager.get_user_count()
    
    def get_all_users(self) -> List[str]:
        """Obtiene la lista de todos los usuarios registrados"""
        return self.dataset_manager.get_all_users()
    
    def delete_user(self, username: str) -> Dict:
        """Elimina un usuario del sistema"""
        try:
            if username not in self.dataset_manager.get_all_users():
                return {
                    "success": False,
                    "message": f"Usuario '{username}' no encontrado."
                }
            
            success = self.dataset_manager.delete_user(username)
            
            if success:
                return {
                    "success": True,
                    "message": f"Usuario '{username}' eliminado exitosamente."
                }
            else:
                return {
                    "success": False,
                    "message": f"Error al eliminar el usuario '{username}'."
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error interno del servidor: {str(e)}"
            }
    
    def delete_all_users(self) -> Dict:
        """
        Elimina todos los usuarios del sistema
        
        Returns:
            Diccionario con el resultado de la eliminación masiva
        """
        try:
            # Obtener lista de todos los usuarios
            all_users = self.dataset_manager.get_all_users()
            
            if not all_users:
                return {
                    "success": True,
                    "message": "No hay usuarios para eliminar.",
                    "deleted_count": 0,
                    "deleted_users": []
                }
            
            # Contar usuarios antes de eliminar
            user_count = len(all_users)
            
            # Eliminar todos los usuarios usando el método del dataset manager
            success = self.dataset_manager.delete_all_users()
            
            if success:
                return {
                    "success": True,
                    "message": f"Todos los usuarios han sido eliminados exitosamente. Total eliminados: {user_count}",
                    "deleted_count": user_count,
                    "deleted_users": all_users
                }
            else:
                return {
                    "success": False,
                    "message": "Error al eliminar todos los usuarios.",
                    "deleted_count": 0,
                    "deleted_users": []
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Error interno del servidor: {str(e)}",
                "deleted_count": 0,
                "deleted_users": []
            }
    
    def get_user_details(self, username: str) -> Dict:
        """
        Obtiene detalles específicos de un usuario
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Diccionario con los detalles del usuario
        """
        try:
            # Verificar si el usuario existe
            if username not in self.dataset_manager.get_all_users():
                return {
                    "success": False,
                    "message": f"Usuario '{username}' no encontrado"
                }
            
            # Obtener información del usuario
            user_data = self.dataset_manager.get_user_data(username)
            
            if user_data is None:
                return {
                    "success": False,
                    "message": f"No se encontraron datos para el usuario '{username}'"
                }
            
            # Obtener embeddings
            user_embeddings = self.dataset_manager.get_user_embeddings(username)
            embedding_count = len(user_embeddings) if user_embeddings is not None else 0
            
            # Calcular estadísticas de embeddings
            embedding_stats = {}
            if user_embeddings is not None and len(user_embeddings) > 0:
                embedding_stats = {
                    "mean": float(np.mean(user_embeddings)),
                    "std": float(np.std(user_embeddings)),
                    "min": float(np.min(user_embeddings)),
                    "max": float(np.max(user_embeddings)),
                    "dimension": int(user_embeddings.shape[1]) if len(user_embeddings.shape) > 1 else len(user_embeddings[0])
                }
            
            return {
                "success": True,
                "username": username,
                "embedding_count": embedding_count,
                "registered_date": user_data.get("registered_date", "Unknown"),
                "image_paths": user_data.get("image_paths", []),
                "embedding_stats": embedding_stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error interno del servidor: {str(e)}"
            }
    
    def get_user_embeddings(self, username: str) -> Dict:
        """
        Obtiene los embeddings de un usuario específico
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Diccionario con los embeddings del usuario
        """
        try:
            # Verificar si el usuario existe
            if username not in self.dataset_manager.get_all_users():
                return {
                    "success": False,
                    "message": f"Usuario '{username}' no encontrado"
                }
            
            # Obtener embeddings
            user_embeddings = self.dataset_manager.get_user_embeddings(username)
            
            if user_embeddings is None:
                return {
                    "success": False,
                    "message": f"No se encontraron embeddings para el usuario '{username}'"
                }
            
            embeddings_list = user_embeddings.tolist()
            embedding_dimension = user_embeddings.shape[1] if len(user_embeddings.shape) > 1 else len(user_embeddings[0])
            
            # Calcular estadísticas
            statistics = {
                "count": len(embeddings_list),
                "dimension": int(embedding_dimension),
                "mean": float(np.mean(user_embeddings)),
                "std": float(np.std(user_embeddings)),
                "min": float(np.min(user_embeddings)),
                "max": float(np.max(user_embeddings))
            }
            
            return {
                "success": True,
                "username": username,
                "embedding_count": len(embeddings_list),
                "embeddings": embeddings_list,
                "embedding_dimension": int(embedding_dimension),
                "statistics": statistics
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error interno del servidor: {str(e)}"
            }
    
    def get_all_users_detailed(self) -> Dict:
        """
        Obtiene información detallada de todos los usuarios
        
        Returns:
            Diccionario con información detallada de todos los usuarios
        """
        try:
            all_users = self.dataset_manager.get_all_users()
            users_detail = []
            total_embeddings = 0
            
            for username in all_users:
                user_details = self.get_user_details(username)
                if user_details.get("success", False):
                    user_detail = {
                        "username": user_details["username"],
                        "embedding_count": user_details["embedding_count"],
                        "registered_date": user_details["registered_date"],
                        "image_paths": user_details["image_paths"],
                        "embedding_stats": user_details.get("embedding_stats", {})
                    }
                    users_detail.append(user_detail)
                    total_embeddings += user_details["embedding_count"]
            
            return {
                "users": users_detail,
                "total_count": len(users_detail),
                "total_embeddings": total_embeddings
            }
            
        except Exception as e:
            return {
                "users": [],
                "total_count": 0,
                "total_embeddings": 0,
                "error": str(e)
            }
    
    def get_health_status(self) -> Dict:
        """Obtiene el estado de salud del sistema"""
        try:
            return {
                "status": "healthy" if self.model_loaded else "unhealthy",
                "model_loaded": self.model_loaded,
                "model_path": self.model_path,
                "user_count": self.get_user_count(),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            return {
                "status": "error",
                "model_loaded": False,
                "model_path": self.model_path,
                "user_count": 0,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }