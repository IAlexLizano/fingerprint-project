"""
Procesador de Datos para Huellas Dactilares
Optimizado para el dataset SOCOFing
"""
import cv2
import os
import json
import random
import numpy as np
from typing import Tuple, List, Dict
from collections import defaultdict
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FingerprintDataProcessor:
    def __init__(self, target_size=(96, 96)):
        """
        Inicializa el procesador de datos
        
        Args:
            target_size: Tamaño objetivo para redimensionar las imágenes
        """
        self.target_size = target_size
        
    def preprocess_image(self, image_path):
        """
        Preprocesamiento optimizado para huellas dactilares
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Imagen preprocesada o None si hay error
        """
        try:
            # Cargar imagen
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                return None
            
            # Redimensionar manteniendo relación de aspecto
            img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Aplicar CLAHE para mejorar contraste local
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            # Filtro gaussiano para reducir ruido
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Realzar bordes con un kernel personalizado
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
            img_enhanced = cv2.filter2D(img, -1, kernel)
            
            # Combinar imagen original con bordes realzados
            img = cv2.addWeighted(img, 0.7, img_enhanced, 0.3, 0)
            img = np.clip(img, 0, 255)
            
            # Normalizar a [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error procesando {image_path}: {e}")
            return None
    
    def load_socofing_dataset(self, dataset_path, max_users=50, max_images_per_user=10):
        """
        Carga el dataset SOCOFing de manera optimizada
        
        Args:
            dataset_path: Ruta del archivo JSON del dataset
            max_users: Número máximo de usuarios a cargar
            max_images_per_user: Número máximo de imágenes por usuario
            
        Returns:
            Tuple con datos de imágenes y mapeo de usuarios
        """
        logger.info(f"Cargando dataset desde: {dataset_path}")
        
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")
        
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        
        # Filtrar usuarios con suficientes imágenes
        valid_users = []
        for user_id, user_data in dataset['subjects'].items():
            if len(user_data['images']) >= 2:  # Mínimo 2 imágenes por usuario
                valid_users.append(user_id)
        
        # Tomar subset de usuarios
        selected_users = valid_users[:max_users]
        
        images_data = []
        user_mapping = {}
        
        for idx, user_id in enumerate(selected_users):
            user_data = dataset['subjects'][user_id]
            user_mapping[user_id] = idx
            
            # Filtrar imágenes válidas
            valid_images = []
            for img_info in user_data['images']:
                if os.path.exists(img_info['path']):
                    valid_images.append(img_info)
            
            # Tomar subset de imágenes
            selected_images = valid_images[:max_images_per_user]
            
            for img_info in selected_images:
                images_data.append({
                    'path': img_info['path'],
                    'user_id': user_id,
                    'user_idx': idx,
                    'finger': img_info.get('finger', 'unknown'),
                    'hand': img_info.get('hand', 'unknown')
                })
        
        logger.info(f"Dataset cargado: {len(selected_users)} usuarios, {len(images_data)} imágenes")
        return images_data, user_mapping
    
    def generate_balanced_pairs(self, images_data, num_pairs=2000):
        """
        Genera pares balanceados para entrenamiento
        
        Args:
            images_data: Lista de datos de imágenes
            num_pairs: Número total de pares a generar
            
        Returns:
            Tuple con pares y etiquetas
        """
        logger.info(f"Generando {num_pairs} pares balanceados...")
        
        # Agrupar por usuario
        user_images = defaultdict(list)
        for img_data in images_data:
            user_images[img_data['user_idx']].append(img_data)
        
        # Filtrar usuarios con al menos 2 imágenes
        valid_users = {k: v for k, v in user_images.items() if len(v) >= 2}
        
        pairs = []
        labels = []
        
        # Generar pares positivos (mismo usuario)
        positive_pairs = []
        for user_idx, user_imgs in valid_users.items():
            for i in range(len(user_imgs)):
                for j in range(i + 1, len(user_imgs)):
                    positive_pairs.append((user_imgs[i], user_imgs[j]))
        
        # Tomar subset de pares positivos
        num_positive = min(num_pairs // 2, len(positive_pairs))
        selected_positive = random.sample(positive_pairs, num_positive)
        
        pairs.extend(selected_positive)
        labels.extend([1] * num_positive)
        
        # Generar pares negativos (diferentes usuarios)
        negative_pairs = []
        users_list = list(valid_users.keys())
        
        for _ in range(num_pairs // 2):
            user1, user2 = random.sample(users_list, 2)
            img1 = random.choice(valid_users[user1])
            img2 = random.choice(valid_users[user2])
            negative_pairs.append((img1, img2))
        
        pairs.extend(negative_pairs)
        labels.extend([0] * len(negative_pairs))
        
        # Mezclar pares
        combined = list(zip(pairs, labels))
        random.shuffle(combined)
        pairs, labels = zip(*combined)
        
        logger.info(f"Pares generados: {num_positive} positivos, {len(negative_pairs)} negativos")
        return list(pairs), list(labels)
    
    def prepare_training_data(self, pairs, labels):
        """
        Prepara los datos para entrenamiento
        
        Args:
            pairs: Lista de pares de imágenes
            labels: Etiquetas correspondientes
            
        Returns:
            Tuple con arrays de imágenes y etiquetas válidas
        """
        logger.info("Preparando datos para entrenamiento...")
        
        images_a, images_b = [], []
        valid_labels = []
        
        for (img1_data, img2_data), label in zip(pairs, labels):
            img1 = self.preprocess_image(img1_data['path'])
            img2 = self.preprocess_image(img2_data['path'])
            
            if img1 is not None and img2 is not None:
                images_a.append(img1)
                images_b.append(img2)
                valid_labels.append(label)
        
        # Convertir a arrays numpy
        images_a = np.array(images_a)[..., np.newaxis]
        images_b = np.array(images_b)[..., np.newaxis]
        valid_labels = np.array(valid_labels, dtype=np.float32)
        
        logger.info(f"Datos preparados: {len(valid_labels)} pares válidos")
        logger.info(f"Balance: {np.mean(valid_labels):.3f} pares positivos")
        
        return images_a, images_b, valid_labels
    
    def preprocess_single_image(self, image_path):
        """
        Preprocesa una sola imagen para predicción
        
        Args:
            image_path: Ruta de la imagen
            
        Returns:
            Imagen preprocesada lista para predicción
        """
        img = self.preprocess_image(image_path)
        if img is not None:
            return np.expand_dims(img, axis=[0, -1])  # Añadir batch y channel dimensions
        return None
    
    def preprocess_image_array(self, image_array):
        """
        Preprocesa un array de imagen directamente (sin cargar desde archivo)
        
        Args:
            image_array: Array numpy de la imagen en escala de grises
            
        Returns:
            Imagen preprocesada o None si hay error
        """
        try:
            if image_array is None:
                return None
            
            # Redimensionar manteniendo relación de aspecto
            img = cv2.resize(image_array, self.target_size, interpolation=cv2.INTER_AREA)
            
            # Aplicar CLAHE para mejorar contraste local
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            
            # Filtro gaussiano para reducir ruido
            img = cv2.GaussianBlur(img, (3, 3), 0)
            
            # Realzar bordes con un kernel personalizado
            kernel = np.array([[-1, -1, -1],
                              [-1,  8, -1],
                              [-1, -1, -1]])
            img_enhanced = cv2.filter2D(img, -1, kernel)
            
            # Combinar imagen original con bordes realzados
            img = cv2.addWeighted(img, 0.7, img_enhanced, 0.3, 0)
            img = np.clip(img, 0, 255)
            
            # Normalizar a [0, 1]
            img = img.astype(np.float32) / 255.0
            
            return img
            
        except Exception as e:
            logger.error(f"Error procesando array de imagen: {e}")
            return None
