"""
Script de prueba para mostrar el preprocesamiento de imágenes del dataset
Muestra el proceso paso a paso para 1-2 huellas dactilares
"""
import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from data.fingerprint_data_processor import FingerprintDataProcessor

def show_preprocessing_steps(image_path, title="Huella Dactilar"):
    """
    Muestra cada paso del preprocesamiento de una imagen
    """
    print(f"\n=== Procesando: {title} ===")
    print(f"Ruta: {image_path}")
    
    # Verificar que existe la imagen
    if not os.path.exists(image_path):
        print(f"❌ Error: No se encuentra la imagen en {image_path}")
        return
    
    # Crear figura para mostrar pasos
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(f'Preprocesamiento: {title}', fontsize=16)
    
    # 1. Imagen original
    img_original = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    axes[0, 0].imshow(img_original, cmap='gray')
    axes[0, 0].set_title('1. Original')
    axes[0, 0].axis('off')
    print(f"Tamaño original: {img_original.shape}")
    
    # 2. Redimensionado
    target_size = (96, 96)  # Tamaño usado en el entrenamiento
    img_resized = cv2.resize(img_original, target_size, interpolation=cv2.INTER_AREA)
    axes[0, 1].imshow(img_resized, cmap='gray')
    axes[0, 1].set_title('2. Redimensionado (96x96)')
    axes[0, 1].axis('off')
    print(f"Tamaño redimensionado: {img_resized.shape}")
    
    # 3. CLAHE (Ecualización adaptativa del histograma)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_resized)
    axes[0, 2].imshow(img_clahe, cmap='gray')
    axes[0, 2].set_title('3. CLAHE (Contraste)')
    axes[0, 2].axis('off')
    
    # 4. Filtro Gaussiano
    img_gaussian = cv2.GaussianBlur(img_clahe, (3, 3), 0)
    axes[0, 3].imshow(img_gaussian, cmap='gray')
    axes[0, 3].set_title('4. Filtro Gaussiano')
    axes[0, 3].axis('off')
    
    # 5. Realce de bordes
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    img_enhanced = cv2.filter2D(img_gaussian, -1, kernel)
    axes[1, 0].imshow(img_enhanced, cmap='gray')
    axes[1, 0].set_title('5. Bordes realzados')
    axes[1, 0].axis('off')
    
    # 6. Combinación de imágenes
    img_combined = cv2.addWeighted(img_gaussian, 0.7, img_enhanced, 0.3, 0)
    img_combined = np.clip(img_combined, 0, 255)
    axes[1, 1].imshow(img_combined, cmap='gray')
    axes[1, 1].set_title('6. Combinación (70% + 30%)')
    axes[1, 1].axis('off')
    
    # 7. Normalización
    img_normalized = img_combined.astype(np.float32) / 255.0
    axes[1, 2].imshow(img_normalized, cmap='gray')
    axes[1, 2].set_title('7. Normalizada [0,1]')
    axes[1, 2].axis('off')
    print(f"Rango después de normalización: [{img_normalized.min():.3f}, {img_normalized.max():.3f}]")
    
    # 8. Forma final para el modelo
    img_final = np.expand_dims(img_normalized, axis=-1)  # Añadir canal
    axes[1, 3].imshow(img_final[:,:,0], cmap='gray')
    axes[1, 3].set_title(f'8. Final: {img_final.shape}')
    axes[1, 3].axis('off')
    print(f"Forma final para el modelo: {img_final.shape}")
    
    plt.tight_layout()
    plt.show()
    
    return img_final

def compare_with_processor():
    """
    Compara el procesamiento manual con el procesador de datos
    """
    print("\n=== Comparando con FingerprintDataProcessor ===")
    
    # Inicializar procesador
    processor = FingerprintDataProcessor(target_size=(96, 96))
    
    # Cargar dataset para obtener rutas de imágenes
    dataset_path = "data/socofing_dataset.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: No se encuentra el dataset en {dataset_path}")
        return
    
    # Cargar algunas imágenes del dataset
    images_data, user_mapping = processor.load_socofing_dataset(
        dataset_path, 
        max_users=2, 
        max_images_per_user=1
    )
    
    if len(images_data) == 0:
        print("❌ Error: No se pudieron cargar imágenes del dataset")
        return
    
    print(f"Imágenes cargadas: {len(images_data)}")
    
    # Procesar primera imagen con el procesador
    first_image = images_data[0]
    print(f"\nProcesando con FingerprintDataProcessor:")
    print(f"Usuario: {first_image['user_idx']}")
    print(f"Ruta: {first_image['path']}")
    
    # Usar el procesador oficial
    processed_img = processor.preprocess_image(first_image['path'])
    
    if processed_img is not None:
        print(f"Resultado del procesador: {processed_img.shape}")
        print(f"Rango: [{processed_img.min():.3f}, {processed_img.max():.3f}]")
        
        # Mostrar resultado
        plt.figure(figsize=(6, 6))
        plt.imshow(processed_img, cmap='gray')
        plt.title('Resultado de FingerprintDataProcessor')
        plt.axis('off')
        plt.show()
    else:
        print("❌ Error: El procesador no pudo procesar la imagen")

def main():
    """
    Función principal para ejecutar las pruebas
    """
    print("🔬 PRUEBA DE PREPROCESAMIENTO DE HUELLAS DACTILARES")
    print("=" * 60)
    
    # Intentar encontrar algunas imágenes del dataset
    dataset_folders = [
        "data/Altered-Easy",
        "data/dataset", 
        "data/DB1_A"
    ]
    
    sample_images = []
    
    # Buscar imágenes de muestra
    for folder in dataset_folders:
        if os.path.exists(folder):
            files = [f for f in os.listdir(folder) if f.lower().endswith(('.bmp', '.jpg', '.png', '.jpeg'))]
            if files:
                sample_images.extend([os.path.join(folder, f) for f in files[:2]])
                break
    
    if not sample_images:
        print("❌ No se encontraron imágenes de muestra en las carpetas del dataset")
        print("Carpetas buscadas:", dataset_folders)
        return
    
    # Mostrar preprocesamiento paso a paso para 1-2 imágenes
    for i, image_path in enumerate(sample_images[:2]):
        title = f"Muestra {i+1}"
        show_preprocessing_steps(image_path, title)
    
    # Comparar con el procesador oficial
    compare_with_processor()
    
    print("\n✅ Prueba de preprocesamiento completada")

if __name__ == "__main__":
    main()
