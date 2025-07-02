#!/usr/bin/env python3
"""
Red Siamesa optimizada específicamente para huellas dactilares
"""
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import os
from typing import Tuple, List

class FingerprintSiameseNetwork:
    def __init__(self, input_shape: Tuple[int, int, int] = (128, 128, 1), embedding_dim: int = 128):
        """
        Red Siamesa balanceada para huellas dactilares
        
        Args:
            input_shape: Forma de las imágenes de entrada
            embedding_dim: Dimensión del embedding (reducida para evitar overfitting)
        """
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.model = self._build_model()
        
    def _build_base_network(self) -> Model:
        input_img = layers.Input(shape=self.input_shape)
        x = input_img
        
        # Bloque 1: Características básicas (32 filtros)
        x = layers.Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Bloque 2: Características intermedias (64 filtros)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.2)(x)
        
        # Bloque 3: Características de alto nivel (128 filtros)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.3)(x)
        
        # Capas densas simplificadas
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        
        # Embedding final (sin normalización L2 que causa problemas)
        embedding = layers.Dense(self.embedding_dim, activation='tanh', name='embedding')(x)
        
        return Model(input_img, embedding, name='fingerprint_encoder')
    
    def _build_model(self) -> Model:
        """Construye el modelo siamesa completo"""
        # Entradas
        input_a = layers.Input(shape=self.input_shape, name='input_a')
        input_b = layers.Input(shape=self.input_shape, name='input_b')
        
        # Red base compartida
        base_network = self._build_base_network()
        
        # Generar embeddings
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        # Calcular similitud usando múltiples métricas
        diff = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([embedding_a, embedding_b])
        mult = layers.Lambda(lambda x: x[0] * x[1])([embedding_a, embedding_b])
        
        # Combinar ambas métricas
        combined = layers.Concatenate()([diff, mult])
        
        # Capas de clasificación simplificadas
        x = layers.Dense(128, activation='relu')(combined)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        output = layers.Dense(1, activation='sigmoid')(x)
        
        model = Model(inputs=[input_a, input_b], outputs=output)
        
        # Compilar con configuración más conservadora
        model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=0.001,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7
            ),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray], 
              validation_data: Tuple[np.ndarray, np.ndarray, np.ndarray] = None,
              epochs: int = 50, batch_size: int = 32, callbacks: list = None) -> dict:
        """Entrena el modelo"""
        pairs_a, pairs_b, labels = train_data
        
        print(f"📊 Verificando datos de entrada:")
        print(f"   - Forma pairs_a: {pairs_a.shape}")
        print(f"   - Forma pairs_b: {pairs_b.shape}")
        print(f"   - Rango valores: [{pairs_a.min():.3f}, {pairs_a.max():.3f}]")
        print(f"   - Balance de clases: {np.mean(labels):.3f}")
        
        if validation_data:
            val_pairs_a, val_pairs_b, val_labels = validation_data
            validation_data = ([val_pairs_a, val_pairs_b], val_labels)
        
        # Añadir callbacks por defecto si no se proporcionan (menos agresivos)
        if callbacks is None:
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,  # Más paciencia (era 5)
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,  # Más paciencia (era 3)
                    min_lr=0.00001,
                    verbose=1
                )
            ]
        
        history = self.model.fit(
            [pairs_a, pairs_b],
            labels,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        return history.history
    
    def save_model(self, model_path: str):
        """Guarda el modelo"""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        print(f"💾 Modelo guardado en: {model_path}")
    
    def load_model(self, model_path: str):
        """Carga el modelo"""
        self.model = tf.keras.models.load_model(model_path)
        print(f"📂 Modelo cargado desde: {model_path}")
    
    def predict_similarity(self, image1: np.ndarray, image2: np.ndarray) -> float:
        """Predice similitud entre dos imágenes"""
        if len(image1.shape) == 3:
            image1 = np.expand_dims(image1, axis=0)
        if len(image2.shape) == 3:
            image2 = np.expand_dims(image2, axis=0)
            
        similarity = self.model.predict([image1, image2], verbose=0)
        return float(similarity[0][0])
    
    def generate_embedding(self, image: np.ndarray) -> np.ndarray:
        """Genera embedding para una imagen"""
        # Buscar el modelo base (encoder) por nombre
        base_network = None
        for layer in self.model.layers:
            if hasattr(layer, 'name') and layer.name == 'fingerprint_encoder':
                base_network = layer
                break
        
        if base_network is None:
            # Si no encontramos la capa por nombre, buscar por tipo de modelo
            for layer in self.model.layers:
                if hasattr(layer, 'layers') and len(layer.layers) > 0:
                    # Verificar si es un modelo funcional que termina con embedding
                    last_layer = layer.layers[-1]
                    if hasattr(last_layer, 'name') and last_layer.name == 'embedding':
                        base_network = layer
                        break
        
        if base_network is None:
            # Como último recurso, crear un modelo desde la primera entrada hasta la capa de embedding
            try:
                # Buscar la capa de embedding en todo el modelo
                embedding_layer = None
                for layer in self.model.layers:
                    if hasattr(layer, 'name') and layer.name == 'embedding':
                        embedding_layer = layer
                        break
                
                if embedding_layer is not None:
                    # Crear un modelo que va desde la entrada hasta la capa de embedding
                    embedding_model = Model(
                        inputs=self.model.input[0],  # Primera entrada del modelo siamés
                        outputs=embedding_layer.output
                    )
                    base_network = embedding_model
                else:
                    raise ValueError("No se pudo encontrar la capa de embedding")
            except Exception as e:
                print(f"Error al crear modelo de embedding: {e}")
                # Crear un modelo extrayendo las características antes de la comparación
                intermediate_model = Model(
                    inputs=self.model.input[0],
                    outputs=self.model.layers[2].output  # Salida del encoder base
                )
                base_network = intermediate_model
        
        # Asegurar que la imagen tenga la forma correcta
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Generar embedding
        embedding = base_network.predict(image, verbose=0)
        
        # Asegurar que el embedding tenga la forma correcta
        if len(embedding.shape) > 1:
            embedding = embedding[0]  # Tomar el primer elemento si es un batch
        
        # Verificar que la dimensión sea correcta
        if len(embedding) != self.embedding_dim:
            print(f"⚠️  Advertencia: Embedding tiene dimensión {len(embedding)}, esperada {self.embedding_dim}")
            # Si la dimensión es incorrecta, redimensionar o truncar
            if len(embedding) > self.embedding_dim:
                embedding = embedding[:self.embedding_dim]
            else:
                # Pad con ceros si es menor
                padded = np.zeros(self.embedding_dim)
                padded[:len(embedding)] = embedding
                embedding = padded
        
        return embedding
    
    def generate_embedding_from_base64(self, image_base64: str) -> np.ndarray:
        """Genera embedding desde imagen en base64"""
        import base64
        import cv2
        from io import BytesIO
        
        try:
            # Decodificar base64
            image_data = base64.b64decode(image_base64)
            
            # Convertir a array de numpy
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
            
            if image is None:
                raise ValueError("No se pudo decodificar la imagen")
            
            # Procesar imagen (mismo preprocesamiento que en entrenamiento)
            image = cv2.resize(image, (128, 128))
            image = image.astype(np.float32) / 255.0
            image = np.expand_dims(image, axis=-1)  # Agregar canal
            
            # Generar embedding
            return self.generate_embedding(image)
            
        except Exception as e:
            raise ValueError(f"Error procesando imagen base64: {str(e)}")
    
    def compare_embeddings(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compara dos embeddings y retorna un score de similitud"""
        # Calcular distancia coseno
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        cosine_similarity = dot_product / (norm1 * norm2)
        
        # Convertir a score de similitud (0-1, donde 1 es más similar)
        similarity_score = (cosine_similarity + 1) / 2
        return float(similarity_score)
