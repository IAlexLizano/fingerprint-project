import os
import numpy as np
from typing import List, Optional, Tuple, Dict
from datetime import datetime

from models.fingerprint_siamese import FingerprintSiameseNetwork
from data.dataset_manager import DatasetManager

class FingerprintService:
    def __init__(self, model_path: str = "models/saved/fingerprint_siamese_model.h5"):
        """
        Inicializa el servicio de autenticación de huellas dactilares
        
        Args:
            model_path: Ruta al modelo entrenado
        """
        self.model_path = model_path
        self.siamese_network = FingerprintSiameseNetwork()
        self.dataset_manager = DatasetManager()
        
        # Cargar modelo si existe
        self.model_loaded = False
        if os.path.exists(model_path):
            try:
                self.siamese_network.load_model(model_path)
                self.model_loaded = True
                print("Modelo cargado exitosamente")
            except Exception as e:
                print(f"Error al cargar el modelo: {e}")
                # Intenta cargar el modelo anterior si existe
                old_model_path = "models/saved/siamese_model.h5"
                if os.path.exists(old_model_path):
                    try:
                        self.siamese_network.load_model(old_model_path)
                        self.model_loaded = True
                        print("Modelo anterior cargado exitosamente")
                    except Exception as e2:
                        print(f"Error al cargar el modelo anterior: {e2}")
    
    def register_user(self, username: str, images_base64: List[str]) -> Dict:
        """
        Registra un nuevo usuario con sus huellas dactilares
        
        Args:
            username: Nombre del usuario
            images_base64: Lista de imágenes en formato base64
            
        Returns:
            Diccionario con el resultado del registro
        """
        try:
            if not self.model_loaded:
                return {
                    "success": False,
                    "message": "Modelo no cargado. Primero debe entrenar el modelo.",
                    "username": None,
                    "embedding_count": None
                }
            
            if len(images_base64) < 1:
                return {
                    "success": False,
                    "message": "Se requiere al menos una imagen para el registro.",
                    "username": None,
                    "embedding_count": None
                }
            
            # Verificar que el usuario no exista
            if username in self.dataset_manager.get_all_users():
                return {
                    "success": False,
                    "message": f"El usuario '{username}' ya existe.",
                    "username": None,
                    "embedding_count": None
                }
            
            # Generar embeddings para cada imagen
            embeddings = []
            for i, image_base64 in enumerate(images_base64):
                try:
                    embedding = self.siamese_network.generate_embedding_from_base64(image_base64)
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
    
    def authenticate_user(self, image_base64: str, threshold: float = 0.5) -> Dict:
        """
        Autentica un usuario basado en su huella dactilar
        
        Args:
            image_base64: Imagen en formato base64
            threshold: Umbral de similitud mínimo
            
        Returns:
            Diccionario con el resultado de la autenticación
        """
        try:
            if not self.model_loaded:
                return {
                    "success": False,
                    "authenticated": False,
                    "username": None,
                    "similarity_score": None,
                    "message": "Modelo no cargado. Primero debe entrenar el modelo."
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
                query_embedding = self.siamese_network.generate_embedding_from_base64(image_base64)
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
                    "similarity_score": float(similarity_score),
                    "message": "No se encontró una coincidencia válida."
                }
                
        except Exception as e:
            return {
                "success": False,
                "authenticated": False,
                "username": None,
                "similarity_score": None,
                "message": f"Error interno del servidor: {str(e)}"
            }
    
    def train_model(self, dataset_path: str, epochs: int = 20, 
                   batch_size: int = 32, validation_split: float = 0.2) -> Dict:
        """
        Entrena el modelo siamesa usando el nuevo enfoque optimizado
        
        Args:
            dataset_path: Ruta al archivo JSON del dataset
            epochs: Número de épocas de entrenamiento (reducido para pruebas más rápidas)
            batch_size: Tamaño del batch
            validation_split: Proporción de datos para validación
            
        Returns:
            Diccionario con el resultado del entrenamiento
        """
        try:
            # Verificar que el archivo de dataset existe
            if not os.path.exists(dataset_path):
                return {
                    "success": False,
                    "message": f"Archivo de dataset no encontrado: {dataset_path}",
                    "training_history": None,
                    "final_accuracy": None,
                    "final_loss": None
                }
            
            print(f"🚀 Iniciando entrenamiento desde el servicio...")
            print(f"   - Dataset: {dataset_path}")
            print(f"   - Épocas: {epochs}, Batch size: {batch_size}")
            
            # Importar las funciones de entrenamiento del script principal
            import sys
            import json
            from multiprocessing import cpu_count
            sys.path.append('.')
            from train_fvc_model import load_dataset_parallel, create_training_pairs
            
            # Cargar datos usando el enfoque optimizado
            num_workers = max(1, cpu_count() // 2)
            images, labels = load_dataset_parallel(dataset_path, num_workers)
            
            # Crear pares de entrenamiento
            pairs_a, pairs_b, pair_labels = create_training_pairs(images, labels)
            
            # Dividir en entrenamiento y validación
            split_idx = int(len(pairs_a) * (1 - validation_split))
            train_a, val_a = pairs_a[:split_idx], pairs_a[split_idx:]
            train_b, val_b = pairs_b[:split_idx], pairs_b[split_idx:]
            train_labels, val_labels = pair_labels[:split_idx], pair_labels[split_idx:]
            
            # Reinicializar la red siamesa para el entrenamiento
            self.siamese_network = FingerprintSiameseNetwork()
            
            # Entrenar el modelo
            history = self.siamese_network.train(
                train_data=(train_a, train_b, train_labels),
                validation_data=(val_a, val_b, val_labels),
                epochs=epochs,
                batch_size=batch_size
            )
            
            # Guardar el modelo entrenado
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            self.siamese_network.save_model(self.model_path)
            self.model_loaded = True
            
            # Obtener métricas finales
            final_accuracy = history.get('accuracy', [0])[-1] if 'accuracy' in history else None
            final_loss = history.get('loss', [0])[-1] if 'loss' in history else None
            
            return {
                "success": True,
                "message": "Modelo entrenado exitosamente con enfoque optimizado",
                "training_history": history,
                "final_accuracy": final_accuracy,
                "final_loss": final_loss
            }
            
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ Error durante el entrenamiento: {error_detail}")
            
            return {
                "success": False,
                "message": f"Error durante el entrenamiento: {str(e)}",
                "training_history": None,
                "final_accuracy": None,
                "final_loss": None
            }
    
    def get_dataset_info(self) -> Dict:
        """
        Obtiene información del dataset
        
        Returns:
            Diccionario con información del dataset
        """
        try:
            return self.dataset_manager.get_dataset_info()
        except Exception as e:
            return {
                "error": str(e),
                "total_users": 0,
                "total_embeddings": 0,
                "users": [],
                "dataset_path": "",
                "embeddings_path": ""
            }
    
    def get_all_users(self) -> List[str]:
        """
        Obtiene la lista de todos los usuarios registrados
        
        Returns:
            Lista de nombres de usuario
        """
        try:
            return self.dataset_manager.get_all_users()
        except Exception as e:
            print(f"Error al obtener usuarios: {e}")
            return []
    
    def delete_user(self, username: str) -> Dict:
        """
        Elimina un usuario del sistema
        
        Args:
            username: Nombre del usuario a eliminar
            
        Returns:
            Diccionario con el resultado de la eliminación
        """
        try:
            success = self.dataset_manager.delete_user(username)
            
            if success:
                return {
                    "success": True,
                    "message": f"Usuario '{username}' eliminado exitosamente."
                }
            else:
                return {
                    "success": False,
                    "message": f"Error al eliminar el usuario '{username}' o el usuario no existe."
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
    
    def get_health_status(self) -> Dict:
        """
        Obtiene el estado de salud del sistema
        
        Returns:
            Diccionario con el estado del sistema
        """
        try:
            dataset_info = self.get_dataset_info()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now(),
                "model_loaded": self.model_loaded,
                "dataset_info": dataset_info
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "timestamp": datetime.now(),
                "model_loaded": self.model_loaded,
                "error": str(e)
            }
    
    def load_model(self) -> bool:
        """
        Carga el modelo entrenado
        
        Returns:
            True si el modelo se cargó exitosamente, False en caso contrario
        """
        try:
            if os.path.exists(self.model_path):
                self.siamese_network.load_model(self.model_path)
                self.model_loaded = True
                return True
            else:
                self.model_loaded = False
                return False
        except Exception as e:
            print(f"Error al cargar el modelo: {e}")
            self.model_loaded = False
            return False
        
    def get_user_details(self, username: str) -> Dict:
        """
        Obtiene información detallada de un usuario específico incluyendo embeddings
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Diccionario con información detallada del usuario
        """
        try:
            # Verificar si el usuario existe
            if username not in self.dataset_manager.get_all_users():
                return {
                    "success": False,
                    "message": f"Usuario '{username}' no encontrado"
                }
            
            # Obtener información básica del usuario
            user_info = self.dataset_manager.get_user_info(username)
            if not user_info:
                return {
                    "success": False,
                    "message": f"No se pudo obtener información del usuario '{username}'"
                }
            
            # Obtener embeddings del usuario
            user_embeddings = self.dataset_manager.get_user_embeddings(username)
            embeddings_list = user_embeddings.tolist() if user_embeddings is not None else []
            
            # Calcular estadísticas de embeddings
            stats = {}
            if user_embeddings is not None and len(user_embeddings) > 0:
                stats = {
                    "mean": float(np.mean(user_embeddings)),
                    "std": float(np.std(user_embeddings)),
                    "min": float(np.min(user_embeddings)),
                    "max": float(np.max(user_embeddings)),
                    "embedding_dimension": int(user_embeddings.shape[1]) if len(user_embeddings.shape) > 1 else int(len(user_embeddings[0]))
                }
            
            return {
                "success": True,
                "username": username,
                "embedding_count": user_info.get("embedding_count", 0),
                "registered_date": user_info.get("registered_date", ""),
                "image_paths": user_info.get("image_paths", []),
                "embeddings": embeddings_list,
                "embedding_stats": stats
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Error interno del servidor: {str(e)}"
            }
    
    def get_user_embeddings(self, username: str) -> Dict:
        """
        Obtiene solo los embeddings de un usuario específico
        
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
                        "embeddings": user_details.get("embeddings", []),
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