#!/usr/bin/env python3
"""
Entrenamiento de red neuronal siamesa para FVC2004 DB1_A
Script limpio y directo para entrenar el modelo de huellas dactilares
"""
import os
import json
import numpy as np
import cv2
from multiprocessing import Pool, cpu_count
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf
from models.fingerprint_siamese import FingerprintSiameseNetwork

def preprocess_image_worker(args):
    """
    Procesa una imagen individual con manejo robusto de rutas
    Args:
        args: tupla (image_path, base_dir) donde base_dir es el directorio raíz del proyecto
    """
    image_path, base_dir = args
    try:
        # Si la ruta es absoluta pero incorrecta, intentar corregirla
        if os.path.isabs(image_path) and not os.path.exists(image_path):
            # Extraer solo el nombre del archivo y reconstruir la ruta
            filename = os.path.basename(image_path)
            corrected_path = os.path.join(base_dir, "data", "DB1_A", filename)
            if os.path.exists(corrected_path):
                image_path = corrected_path
            else:
                print(f"⚠️  Archivo no encontrado: {filename}")
                return None
        
        # Si la ruta es relativa, convertirla a absoluta
        elif not os.path.isabs(image_path):
            image_path = os.path.join(base_dir, image_path)
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"⚠️  No se pudo cargar la imagen: {image_path}")
            return None
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.resize(image, (128, 128))
        image = image.astype(np.float32) / 255.0
        image = np.expand_dims(image, axis=-1)
        return image
    except Exception as e:
        print(f"⚠️  Error procesando {image_path}: {e}")
        return None

def load_dataset_parallel(dataset_path, num_workers, base_dir):
    print(f"🔄 Cargando dataset con {num_workers} procesos...")
    print(f"📁 Directorio base: {base_dir}")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    dataset = dataset[:30]
    print(f"🧪 Usando {len(dataset)} usuarios para entrenamiento")
    
    all_image_paths = []
    labels = []
    
    for user_data in dataset:
        username = user_data['username']
        for image_path in user_data['image_paths']:
            all_image_paths.append(image_path)
            labels.append(username)
    
    print(f"📊 Total de rutas de imágenes encontradas: {len(all_image_paths)}")
    
    # Preparar argumentos para el worker con el directorio base
    worker_args = [(path, base_dir) for path in all_image_paths]
    
    with Pool(processes=num_workers) as pool:
        images = pool.map(preprocess_image_worker, worker_args)
    
    processed_images = []
    valid_labels = []
    invalid_count = 0
    
    for img, label in zip(images, labels):
        if img is not None:
            processed_images.append(img)
            valid_labels.append(label)
        else:
            invalid_count += 1
    
    print(f"✅ Imágenes procesadas: {len(processed_images)}/{len(all_image_paths)}")
    if invalid_count > 0:
        print(f"⚠️  Imágenes no válidas: {invalid_count}")
    
    return processed_images, valid_labels

def create_training_pairs(images, labels, positive_pairs_ratio=0.5):
    print("🔗 Creando pares de entrenamiento...")
    user_images = {}
    for i, label in enumerate(labels):
        if label not in user_images:
            user_images[label] = []
        user_images[label].append(images[i])
    positive_pairs_a, positive_pairs_b = [], []
    for imgs in user_images.values():
        n = len(imgs)
        for i in range(n):
            for j in range(i+1, n):
                positive_pairs_a.append(imgs[i])
                positive_pairs_b.append(imgs[j])
    negative_pairs_a, negative_pairs_b = [], []
    user_list = list(user_images.keys())
    n_neg = int(len(positive_pairs_a) * (1 - positive_pairs_ratio) / positive_pairs_ratio)
    for _ in range(n_neg):
        u1, u2 = random.sample(user_list, 2)
        img1 = random.choice(user_images[u1])
        img2 = random.choice(user_images[u2])
        negative_pairs_a.append(img1)
        negative_pairs_b.append(img2)
    pairs_a = positive_pairs_a + negative_pairs_a
    pairs_b = positive_pairs_b + negative_pairs_b
    pair_labels = [1]*len(positive_pairs_a) + [0]*len(negative_pairs_a)
    idx = np.random.permutation(len(pairs_a))
    pairs_a = [pairs_a[i] for i in idx]
    pairs_b = [pairs_b[i] for i in idx]
    pair_labels = [pair_labels[i] for i in idx]
    print(f"✅ Total pares: {len(pairs_a)} (Positivos: {sum(pair_labels)}, Negativos: {len(pair_labels)-sum(pair_labels)})")
    return np.array(pairs_a), np.array(pairs_b), np.array(pair_labels, dtype=np.float32)

def plot_metrics(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='Pérdida entrenamiento')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Pérdida validación')
    plt.title('Pérdida')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='Precisión entrenamiento')
    if 'val_accuracy' in history:
        plt.plot(history['val_accuracy'], label='Precisión validación')
    plt.title('Precisión')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Diferente", "Misma persona"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(title)
    plt.show()

def train_siamese_model(train_a, train_b, train_labels, val_a, val_b, val_labels, epochs, batch_size):
    print("🤖 Inicializando red neuronal siamesa optimizada...")
    siamese = FingerprintSiameseNetwork()
    
    print("Datos de entrenamiento:")
    print(f"  - Pares de entrenamiento: {len(train_a)}")
    print(f"  - Pares de validación: {len(val_a)}")
    print(f"  - Pares positivos entrenamiento: {sum(train_labels)}")
    print(f"  - Pares negativos entrenamiento: {len(train_labels) - sum(train_labels)}")
    
    # Verificar que las formas de los datos sean correctas
    print(f"  - Forma train_a: {train_a.shape}")
    print(f"  - Forma train_b: {train_b.shape}")
    print(f"  - Rango de valores: [{train_a.min():.3f}, {train_a.max():.3f}]")
    
    print("🚀 Iniciando entrenamiento...")
    
    # Entrenar el modelo (incluye callbacks por defecto)
    history = siamese.train(
        train_data=(train_a, train_b, train_labels),
        validation_data=(val_a, val_b, val_labels),
        epochs=epochs,
        batch_size=batch_size
    )
    
    # Guardar el modelo usando ruta absoluta
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "models", "saved", "fingerprint_siamese_model.h5")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    siamese.save_model(model_path)
    
    print(f"💾 Modelo guardado en: {model_path}")
    
    return siamese, history

def main():
    # Usar rutas absolutas para evitar problemas de directorios
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = os.path.join(current_dir, "data", "fvc2004_db1a_dataset.json")
    
    epochs = 20  # Más épocas para mejor convergencia
    batch_size = 64  # Batch size estándar
    validation_split = 0.2
    num_workers = max(1, cpu_count() // 2)
    max_users = 25  # Más usuarios para mejor generalización
    print(f"🧪 ENTRENAMIENTO MEJORADO: {num_workers} procesos, {epochs} épocas, batch_size={batch_size}, max_users={max_users}")
    print(f"📁 Directorio de trabajo: {current_dir}")
    print(f"📄 Ruta del dataset: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Dataset no encontrado en {dataset_path}")
        return
    
    # Cargar y procesar más usuarios
    images, labels = load_dataset_parallel(dataset_path, num_workers, current_dir)
    
    print(f"📊 Estadísticas del dataset:")
    print(f"   - Total imágenes: {len(images)}")
    print(f"   - Usuarios únicos: {len(set(labels))}")
    print(f"   - Imágenes por usuario: {len(images) / len(set(labels)):.1f}")
    
    # Crear pares con mejor balance
    pairs_a, pairs_b, pair_labels = create_training_pairs(images, labels, positive_pairs_ratio=0.5)
    
    # Verificar balance de datos
    positive_ratio = sum(pair_labels) / len(pair_labels)
    print(f"   - Ratio pares positivos: {positive_ratio:.3f}")
    
    # Mezclar datos antes de dividir
    indices = np.random.permutation(len(pairs_a))
    pairs_a = pairs_a[indices]
    pairs_b = pairs_b[indices]
    pair_labels = pair_labels[indices]
    
    split_idx = int(len(pairs_a) * (1 - validation_split))
    train_a, val_a = pairs_a[:split_idx], pairs_a[split_idx:]
    train_b, val_b = pairs_b[:split_idx], pairs_b[split_idx:]
    train_labels, val_labels = pair_labels[:split_idx], pair_labels[split_idx:]
    
    print(f"📊 División de datos:")
    print(f"   - Entrenamiento: {len(train_a)} pares")
    print(f"   - Validación: {len(val_a)} pares")
    print(f"   - Balance entrenamiento: {np.mean(train_labels):.3f}")
    print(f"   - Balance validación: {np.mean(val_labels):.3f}")
    
    # Entrenar la red siamesa directamente
    print("🚀 Entrenando modelo...")
    siamese_model, history = train_siamese_model(train_a, train_b, train_labels, 
                                                val_a, val_b, val_labels, epochs, batch_size)
    
    if history and 'accuracy' in history:
        plot_metrics(history)
        print("\n🔎 Matriz de confusión (entrenamiento):")
        train_preds = (siamese_model.model.predict([train_a, train_b], batch_size=batch_size) > 0.5).astype(int).flatten()
        plot_confusion(train_labels, train_preds, "Entrenamiento")
        print("\n🔎 Matriz de confusión (validación):")
        val_preds = (siamese_model.model.predict([val_a, val_b], batch_size=batch_size) > 0.5).astype(int).flatten()
        plot_confusion(val_labels, val_preds, "Validación")
    print("🎉 ¡Entrenamiento completado!")
    # La ruta se imprime desde la función train_siamese_model

if __name__ == "__main__":
    main()
