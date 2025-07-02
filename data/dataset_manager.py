import numpy as np
import json
import os
from typing import List, Dict, Tuple, Optional
import cv2
from PIL import Image
import base64
from io import BytesIO

class DatasetManager:
    def __init__(self, dataset_path: str = "data/dataset", embeddings_path: str = "data/embeddings"):
        """
        Inicializa el gestor de dataset para huellas dactilares
        
        Args:
            dataset_path: Ruta donde se guardan las imágenes del dataset
            embeddings_path: Ruta donde se guardan los embeddings
        """
        self.dataset_path = dataset_path
        self.embeddings_path = embeddings_path
        self.embeddings_file = os.path.join(embeddings_path, "user_embeddings.npy")
        self.users_file = os.path.join(embeddings_path, "users.json")
        
        # Crear directorios si no existen
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(embeddings_path, exist_ok=True)
        
        # Cargar datos existentes
        self.users = self._load_users()
        self.embeddings = self._load_embeddings()
    
    def _load_users(self) -> Dict:
        """Carga la información de usuarios desde el archivo JSON"""
        if os.path.exists(self.users_file):
            with open(self.users_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_users(self):
        """Guarda la información de usuarios en el archivo JSON"""
        with open(self.users_file, 'w') as f:
            json.dump(self.users, f, indent=2)
    
    def _load_embeddings(self) -> np.ndarray:
        """Carga los embeddings desde el archivo .npy"""
        if os.path.exists(self.embeddings_file):
            return np.load(self.embeddings_file)
        return np.array([])
    
    def _save_embeddings(self):
        """Guarda los embeddings en el archivo .npy"""
        np.save(self.embeddings_file, self.embeddings)
    
    def save_base64_image(self, base64_string: str, filename: str) -> str:
        """
        Guarda una imagen en formato base64 como archivo
        
        Args:
            base64_string: Imagen en formato base64
            filename: Nombre del archivo
            
        Returns:
            Ruta del archivo guardado
        """
        # Decodificar base64
        image_data = base64.b64decode(base64_string)
        
        # Guardar imagen
        file_path = os.path.join(self.dataset_path, filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        
        return file_path
    
    def load_image_from_path(self, image_path: str) -> np.ndarray:
        """
        Carga una imagen desde una ruta de archivo
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Imagen como numpy array
        """
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return None
    
    def load_image_from_base64(self, base64_string: str) -> np.ndarray:
        """
        Carga una imagen desde formato base64
        
        Args:
            base64_string: Imagen en formato base64
            
        Returns:
            Imagen como numpy array
        """
        try:
            # Decodificar base64
            image_data = base64.b64decode(base64_string)
            image = Image.open(BytesIO(image_data))
            
            # Convertir a numpy array
            image_array = np.array(image)
            
            return image_array
        except Exception as e:
            print(f"Error al cargar imagen base64: {e}")
            return None
    
    def register_user(self, username: str, images_base64: List[str], 
                     embeddings: List[np.ndarray]) -> bool:
        """
        Registra un nuevo usuario con sus imágenes y embeddings
        
        Args:
            username: Nombre del usuario
            images_base64: Lista de imágenes en formato base64
            embeddings: Lista de embeddings correspondientes
            
        Returns:
            True si el registro fue exitoso, False en caso contrario
        """
        if username in self.users:
            print(f"Usuario {username} ya existe")
            return False
        
        # Guardar imágenes
        image_paths = []
        for i, image_base64 in enumerate(images_base64):
            filename = f"{username}_{i+1}.jpg"
            image_path = self.save_base64_image(image_base64, filename)
            image_paths.append(image_path)
        
        # Guardar información del usuario
        user_info = {
            "username": username,
            "image_paths": image_paths,
            "embedding_count": len(embeddings),
            "registered_date": str(np.datetime64('now'))
        }
        
        self.users[username] = user_info
        
        # Agregar embeddings al array
        embeddings_array = np.array(embeddings)
        
        # Verificar dimensiones de embeddings
        if len(embeddings_array.shape) == 1:
            embeddings_array = embeddings_array.reshape(1, -1)
        
        print(f"📊 Forma de embeddings nuevos: {embeddings_array.shape}")
        
        if len(self.embeddings) == 0:
            self.embeddings = embeddings_array
            print(f"📊 Inicializando embeddings con forma: {self.embeddings.shape}")
        else:
            print(f"📊 Embeddings existentes: {self.embeddings.shape}")
            
            # Verificar compatibilidad de dimensiones
            if self.embeddings.shape[1] != embeddings_array.shape[1]:
                raise ValueError(
                    f"Incompatibilidad de dimensiones: embeddings existentes tienen "
                    f"{self.embeddings.shape[1]} dimensiones, nuevos embeddings tienen "
                    f"{embeddings_array.shape[1]} dimensiones"
                )
            
            self.embeddings = np.vstack([self.embeddings, embeddings_array])
            print(f"📊 Embeddings actualizados: {self.embeddings.shape}")
        
        # Guardar datos
        self._save_users()
        self._save_embeddings()
        
        print(f"Usuario {username} registrado exitosamente con {len(embeddings)} embeddings")
        return True
    
    def get_user_embeddings(self, username: str) -> Optional[np.ndarray]:
        """
        Obtiene los embeddings de un usuario específico
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Array de embeddings del usuario o None si no existe
        """
        if username not in self.users:
            return None
        
        user_info = self.users[username]
        start_idx = self._get_user_start_index(username)
        end_idx = start_idx + user_info["embedding_count"]
        
        return self.embeddings[start_idx:end_idx]
    
    def _get_user_start_index(self, username: str) -> int:
        """
        Obtiene el índice de inicio de los embeddings de un usuario
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Índice de inicio
        """
        start_idx = 0
        for user in self.users:
            if user == username:
                break
            start_idx += self.users[user]["embedding_count"]
        return start_idx
    
    def get_all_users(self) -> List[str]:
        """
        Obtiene la lista de todos los usuarios registrados
        
        Returns:
            Lista de nombres de usuario
        """
        return list(self.users.keys())
    
    def get_user_count(self) -> int:
        """
        Obtiene el número total de usuarios registrados
        
        Returns:
            Número de usuarios
        """
        return len(self.users)
    
    def get_total_embeddings(self) -> int:
        """
        Obtiene el número total de embeddings almacenados
        
        Returns:
            Número total de embeddings
        """
        return len(self.embeddings)
    
    def delete_user(self, username: str) -> bool:
        """
        Elimina un usuario y sus datos asociados
        
        Args:
            username: Nombre del usuario a eliminar
            
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        if username not in self.users:
            print(f"Usuario {username} no existe")
            return False
        
        user_info = self.users[username]
        
        # Eliminar archivos de imagen
        for image_path in user_info["image_paths"]:
            if os.path.exists(image_path):
                os.remove(image_path)
        
        # Eliminar embeddings del array
        start_idx = self._get_user_start_index(username)
        end_idx = start_idx + user_info["embedding_count"]
        
        self.embeddings = np.delete(self.embeddings, 
                                   range(start_idx, end_idx), axis=0)
        
        # Eliminar información del usuario
        del self.users[username]
        
        # Guardar datos actualizados
        self._save_users()
        self._save_embeddings()
        
        print(f"Usuario {username} eliminado exitosamente")
        return True
    
    def delete_all_users(self) -> bool:
        """
        Elimina todos los usuarios y sus datos asociados
        
        Returns:
            True si la eliminación fue exitosa, False en caso contrario
        """
        try:
            print("🗑️  Iniciando eliminación de todos los usuarios...")
            
            # Eliminar todas las imágenes de todos los usuarios
            deleted_images = 0
            for username, user_info in self.users.items():
                for image_path in user_info.get("image_paths", []):
                    if os.path.exists(image_path):
                        try:
                            os.remove(image_path)
                            deleted_images += 1
                        except Exception as e:
                            print(f"⚠️  Error eliminando imagen {image_path}: {e}")
            
            # Limpiar todos los datos
            self.users.clear()
            self.embeddings = np.array([])
            
            # Guardar archivos vacíos
            self._save_users()
            self._save_embeddings()
            
            print(f"✅ Eliminación completa:")
            print(f"   - Usuarios eliminados: {len(self.users)}")
            print(f"   - Imágenes eliminadas: {deleted_images}")
            print(f"   - Embeddings eliminados: Todos")
            
            return True
            
        except Exception as e:
            print(f"❌ Error durante la eliminación masiva: {e}")
            return False

    def find_best_match(self, query_embedding: np.ndarray, 
                       threshold: float = 0.5) -> Tuple[Optional[str], float]:
        """
        Encuentra el mejor match para un embedding de consulta
        
        Args:
            query_embedding: Embedding de la imagen de consulta
            threshold: Umbral de similitud mínimo
            
        Returns:
            Tupla (username, similarity_score) o (None, 0.0) si no hay match
        """
        if len(self.embeddings) == 0:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        # Calcular similitud con todos los embeddings
        for username in self.users:
            user_embeddings = self.get_user_embeddings(username)
            if user_embeddings is None:
                continue
            
            # Calcular similitud con cada embedding del usuario
            for embedding in user_embeddings:
                # Calcular distancia euclidiana
                distance = np.linalg.norm(query_embedding - embedding)
                # Convertir distancia a similitud (menor distancia = mayor similitud)
                similarity = 1.0 / (1.0 + distance)
                
                if similarity > best_score and similarity >= threshold:
                    best_score = similarity
                    best_match = username
        
        return best_match, best_score
    
    def get_dataset_info(self) -> Dict:
        """
        Obtiene información general del dataset
        
        Returns:
            Diccionario con información del dataset
        """
        return {
            "total_users": self.get_user_count(),
            "total_embeddings": self.get_total_embeddings(),
            "users": list(self.users.keys()),
            "dataset_path": self.dataset_path,
            "embeddings_path": self.embeddings_path
        }
    
    def get_user_info(self, username: str) -> Optional[Dict]:
        """
        Obtiene la información completa de un usuario específico
        
        Args:
            username: Nombre del usuario
            
        Returns:
            Diccionario con la información del usuario o None si no existe
        """
        if username not in self.users:
            return None
        
        return self.users[username].copy()