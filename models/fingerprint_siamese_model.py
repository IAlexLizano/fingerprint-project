"""
Modelo Siamesa Mejorado para Reconocimiento de Huellas Dactilares
Arquitectura optimizada para el dataset SOCOFing
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
from typing import Tuple

class ImprovedSiameseNetwork:
    def __init__(self, input_shape: Tuple[int, int, int] = (96, 96, 1), margin=1.0):
        """
        Red Siamesa mejorada con arquitectura optimizada para huellas dactilares
        
        Args:
            input_shape: Dimensiones de entrada (altura, anchura, canales)
            margin: Margen para la contrastive loss
        """
        self.input_shape = input_shape
        self.margin = margin
        self.model = None
        self.base_network = None
        self._build_model()
        
    def _build_base_network(self) -> Model:
        """
        Arquitectura base optimizada para características de huellas dactilares
        """
        input_img = layers.Input(shape=self.input_shape)
        
        # Normalización mejorada
        x = layers.Lambda(lambda x: (x - 0.5) * 2.0)(input_img)
        
        # Bloque 1: Detección de líneas y patrones básicos
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same', 
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.1)(x)
        
        # Bloque 2: Patrones más complejos
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.15)(x)
        
        # Bloque 3: Características de alto nivel
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Bloque 4: Características más específicas
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same',
                         kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Capas densas para embedding
        x = layers.Dense(512, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        
        # Embedding final con normalización L2
        embedding = layers.Dense(128, activation='linear', kernel_initializer='he_normal')(x)
        embedding = layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embedding)
        
        return Model(input_img, embedding, name='fingerprint_encoder')
    
    def contrastive_loss(self, y_true, y_pred):
        """
        Contrastive loss mejorada con mejor estabilidad numérica
        
        Args:
            y_true: Labels (1 para mismo dedo, 0 para dedos diferentes)
            y_pred: Distancia euclidiana predicha
        """
        # Estabilidad numérica
        y_pred = tf.maximum(y_pred, 1e-8)
        
        # Pérdida para pares similares (minimizar distancia)
        loss_pos = y_true * tf.square(y_pred)
        
        # Pérdida para pares diferentes (maximizar distancia hasta el margen)
        loss_neg = (1 - y_true) * tf.square(tf.maximum(0.0, self.margin - y_pred))
        
        return tf.reduce_mean(0.5 * (loss_pos + loss_neg))
    
    def accuracy_metric(self, y_true, y_pred):
        """
        Métrica de precisión basada en threshold adaptativo
        """
        threshold = self.margin / 2.0
        predictions = tf.cast(y_pred < threshold, tf.float32)
        return tf.reduce_mean(tf.cast(tf.equal(predictions, y_true), tf.float32))
    
    def _build_model(self):
        """
        Construye el modelo siamesa completo
        """
        # Crear red base
        self.base_network = self._build_base_network()
        
        # Entradas para los pares
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')
        
        # Generar embeddings
        embedding_a = self.base_network(input_a)
        embedding_b = self.base_network(input_b)
        
        # Calcular distancia euclidiana
        distance = layers.Lambda(
            lambda embeddings: tf.sqrt(tf.reduce_sum(tf.square(embeddings[0] - embeddings[1]), axis=1)),
            name='euclidean_distance'
        )([embedding_a, embedding_b])
        
        # Crear modelo
        self.model = Model(inputs=[input_a, input_b], outputs=distance)
        
        # Compilar con optimizer mejorado
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=self.contrastive_loss,
            metrics=[self.accuracy_metric]
        )
    
    def get_embeddings(self, images):
        """
        Obtiene embeddings para un batch de imágenes
        
        Args:
            images: Array de imágenes
            
        Returns:
            Array de embeddings normalizados
        """
        if len(images.shape) == 3:
            images = np.expand_dims(images, axis=0)
        return self.base_network.predict(images, verbose=0)
    
    def predict_similarity(self, image1, image2):
        """
        Predice similitud entre dos imágenes
        
        Args:
            image1: Primera imagen
            image2: Segunda imagen
            
        Returns:
            Valor de similitud entre 0 y 1
        """
        if len(image1.shape) == 3:
            image1 = np.expand_dims(image1, axis=0)
        if len(image2.shape) == 3:
            image2 = np.expand_dims(image2, axis=0)
        
        distance = self.model.predict([image1, image2], verbose=0)
        # Convertir distancia a similitud (0-1)
        similarity = 1.0 / (1.0 + distance[0])
        return float(similarity)
    
    def save_model(self, filepath):
        """
        Guarda el modelo completo
        """
        self.model.save(filepath)
    
    def save_base_network(self, filepath):
        """
        Guarda solo la red base para extraer embeddings
        """
        self.base_network.save(filepath)
    
    @classmethod
    def load_model(cls, filepath, margin=1.0):
        """
        Carga un modelo guardado
        
        Args:
            filepath: Ruta del modelo guardado
            margin: Margen para la contrastive loss
            
        Returns:
            Instancia del modelo cargado
        """
        instance = cls(margin=margin)
        instance.model = tf.keras.models.load_model(filepath, compile=False)
        
        # Recompilar con las funciones personalizadas
        instance.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss=instance.contrastive_loss,
            metrics=[instance.accuracy_metric]
        )
        
        return instance
