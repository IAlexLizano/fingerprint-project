#!/usr/bin/env python3
"""
Script para depurar el problema de dimensiones de embeddings
"""
import os
import sys
import numpy as np
import base64
import cv2
from models.fingerprint_siamese import FingerprintSiameseNetwork

def create_test_image_base64():
    """Crear una imagen de prueba en base64"""
    # Crear una imagen de prueba (128x128 con patrón de huellas simulado)
    image = np.random.randint(0, 255, (128, 128), dtype=np.uint8)
    
    # Agregar algunos patrones que simulen líneas de huellas
    for i in range(10, 118, 10):
        image[i:i+2, :] = 200  # Líneas horizontales
    
    # Codificar como JPEG y luego a base64
    _, buffer = cv2.imencode('.jpg', image)
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    return image_base64

def test_embedding_generation():
    print("🔬 DEPURACIÓN: Generación de Embeddings")
    print("=" * 50)
    
    # Verificar si existe un modelo entrenado
    model_path = "models/saved/fingerprint_siamese_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en: {model_path}")
        print("🔧 Primero debe entrenar el modelo ejecutando: python train_fvc_model.py")
        return
    
    try:
        # Inicializar la red siamesa
        print("🤖 Inicializando red siamesa...")
        siamese = FingerprintSiameseNetwork()
        
        print(f"📏 Dimensión de embedding configurada: {siamese.embedding_dim}")
        
        # Cargar el modelo
        print(f"📂 Cargando modelo desde: {model_path}")
        siamese.load_model(model_path)
        
        # Mostrar información del modelo
        print(f"📊 Resumen del modelo:")
        siamese.model.summary()
        
        # Crear imagen de prueba
        print("\n🖼️  Creando imagen de prueba...")
        test_image_base64 = create_test_image_base64()
        
        # Generar embedding
        print("🧮 Generando embedding...")
        embedding = siamese.generate_embedding_from_base64(test_image_base64)
        
        print(f"📏 Dimensión del embedding generado: {len(embedding)}")
        print(f"📏 Forma del embedding: {embedding.shape}")
        print(f"📊 Rango de valores: [{embedding.min():.4f}, {embedding.max():.4f}]")
        print(f"📊 Tipo de datos: {embedding.dtype}")
        
        # Verificar si la dimensión es correcta
        if len(embedding) == siamese.embedding_dim:
            print("✅ Dimensión del embedding es correcta")
        else:
            print(f"⚠️  PROBLEMA: Dimensión incorrecta. Esperada: {siamese.embedding_dim}, Obtenida: {len(embedding)}")
        
        # Generar múltiples embeddings para verificar consistencia
        print("\n🔄 Probando consistencia con múltiples embeddings...")
        embeddings = []
        for i in range(3):
            test_img = create_test_image_base64()
            emb = siamese.generate_embedding_from_base64(test_img)
            embeddings.append(emb)
            print(f"   Embedding {i+1}: dimensión {len(emb)}")
        
        # Verificar si todas tienen la misma dimensión
        dimensions = [len(emb) for emb in embeddings]
        if len(set(dimensions)) == 1:
            print("✅ Todos los embeddings tienen la misma dimensión")
        else:
            print(f"❌ PROBLEMA: Dimensiones inconsistentes: {dimensions}")
        
        # Probar concatenación
        try:
            combined = np.vstack(embeddings)
            print(f"✅ Concatenación exitosa: forma {combined.shape}")
        except Exception as e:
            print(f"❌ Error en concatenación: {e}")
        
    except Exception as e:
        print(f"❌ Error durante la depuración: {e}")
        import traceback
        traceback.print_exc()

def check_model_architecture():
    """Verificar la arquitectura del modelo para entender las capas"""
    print("\n🏗️  VERIFICACIÓN DE ARQUITECTURA")
    print("=" * 50)
    
    model_path = "models/saved/fingerprint_siamese_model.h5"
    if not os.path.exists(model_path):
        print(f"❌ No se encontró el modelo en: {model_path}")
        return
    
    try:
        siamese = FingerprintSiameseNetwork()
        siamese.load_model(model_path)
        
        print("📋 Capas del modelo:")
        for i, layer in enumerate(siamese.model.layers):
            layer_name = getattr(layer, 'name', 'Sin nombre')
            layer_type = type(layer).__name__
            
            if hasattr(layer, 'output_shape'):
                output_shape = layer.output_shape
            else:
                output_shape = "Desconocida"
            
            print(f"   {i:2d}: {layer_name:20s} [{layer_type:15s}] -> {output_shape}")
            
            # Buscar capas específicas
            if 'embedding' in layer_name.lower():
                print(f"       ⭐ CAPA DE EMBEDDING ENCONTRADA")
            elif 'fingerprint_encoder' in layer_name.lower():
                print(f"       🎯 ENCODER ENCONTRADO")
    
    except Exception as e:
        print(f"❌ Error al verificar arquitectura: {e}")

if __name__ == "__main__":
    print("🚀 INICIANDO DEPURACIÓN DEL SISTEMA DE EMBEDDINGS")
    print("=" * 60)
    
    # Cambiar al directorio del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(current_dir)
    
    check_model_architecture()
    test_embedding_generation()
    
    print("\n🎉 Depuración completada")
