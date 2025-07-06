"""
Entrenamiento del Modelo Siamesa para Huellas Dactilares
Con visualización de métricas y análisis completo
"""
import tensorflow as tf
import numpy as np
import random
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os
import sys

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.fingerprint_siamese_model import ImprovedSiameseNetwork
from data.fingerprint_data_processor import FingerprintDataProcessor

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurar matplotlib para mostrar gráficas
import matplotlib
matplotlib.use('TkAgg')  # Para mostrar ventanas en Windows

class TrainingVisualizer:
    """Clase para visualizar métricas de entrenamiento"""
    
    def __init__(self, save_dir="training_plots"):
        """
        Inicializa el visualizador
        
        Args:
            save_dir: Directorio donde guardar las gráficas
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_training_history(self, history):
        """
        Muestra las gráficas de entrenamiento en pantalla
        """
        # Crear figura con subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Gráfica de pérdida
        ax1.plot(history.history['loss'], label='Pérdida Entrenamiento', color='blue')
        ax1.plot(history.history['val_loss'], label='Pérdida Validación', color='red')
        ax1.set_title('Pérdida del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de precisión
        if 'accuracy_metric' in history.history:
            ax2.plot(history.history['accuracy_metric'], label='Precisión Entrenamiento', color='blue')
            ax2.plot(history.history['val_accuracy_metric'], label='Precisión Validación', color='red')
        elif 'acc' in history.history:
            ax2.plot(history.history['acc'], label='Precisión Entrenamiento', color='blue')
            ax2.plot(history.history['val_acc'], label='Precisión Validación', color='red')
        
        ax2.set_title('Precisión del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Gráficas de entrenamiento mostradas")
    
    def plot_confusion_matrix(self, y_true, y_pred, threshold=0.5):
        """
        Muestra la matriz de confusión en pantalla
        """
        # Convertir distancias a predicciones binarias
        y_pred_binary = (y_pred < threshold).astype(int)
        
        # Calcular matriz de confusión
        cm = confusion_matrix(y_true, y_pred_binary)
        
        # Crear gráfica
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Diferentes', 'Mismo Usuario'],
                   yticklabels=['Diferentes', 'Mismo Usuario'])
        plt.title('Matriz de Confusión')
        plt.xlabel('Predicción')
        plt.ylabel('Etiqueta Verdadera')
        plt.show()
        
        # Imprimir reporte de clasificación
        logger.info("Reporte de Clasificación:")
        logger.info(f"\n{classification_report(y_true, y_pred_binary, target_names=['Diferentes', 'Mismo Usuario'])}")
    
    def plot_distance_distribution(self, distances_same, distances_diff, threshold=0.5):
        """
        Muestra la distribución de distancias en pantalla
        """
        plt.figure(figsize=(10, 6))
        
        # Histogramas
        plt.hist(distances_same, bins=50, alpha=0.7, label='Mismo Usuario', color='green', density=True)
        plt.hist(distances_diff, bins=50, alpha=0.7, label='Diferentes Usuarios', color='red', density=True)
        
        # Línea del umbral
        plt.axvline(threshold, color='black', linestyle='--', label=f'Umbral ({threshold})')
        
        plt.xlabel('Distancia Euclidiana')
        plt.ylabel('Densidad')
        plt.title('Distribución de Distancias')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        logger.info("Distribución de distancias mostrada")

def train_fingerprint_model():
    """
    Función principal para entrenar el modelo con visualizaciones
    """
    logger.info("=== ENTRENAMIENTO MODELO SIAMESA PARA HUELLAS DACTILARES ===")
    
    # Configuración
    dataset_path = r"C:\development\septimo\ia\projects\final_project\project-fingerprint\data\socofing_dataset.json"
    
    # Inicializar visualizador (sin guardar archivos)
    visualizer = TrainingVisualizer()
    
    # Inicializar procesador de datos
    processor = FingerprintDataProcessor(target_size=(96, 96))
    
    # Cargar dataset
    images_data, user_mapping = processor.load_socofing_dataset(
        dataset_path, 
        max_users=80, 
        max_images_per_user=8
    )
    
    if len(images_data) < 100:
        logger.error("Dataset muy pequeño. Se necesitan al menos 100 imágenes.")
        return
    
    # Generar pares
    pairs, labels = processor.generate_balanced_pairs(images_data, num_pairs=2500)
    
    # Preparar datos
    images_a, images_b, labels = processor.prepare_training_data(pairs, labels)
    
    if len(labels) < 100:
        logger.error("Muy pocos pares válidos para entrenamiento.")
        return
    
    # Dividir en entrenamiento y validación
    split_idx = int(0.85 * len(labels))
    
    train_a, val_a = images_a[:split_idx], images_a[split_idx:]
    train_b, val_b = images_b[:split_idx], images_b[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    logger.info(f"División: {len(train_labels)} entrenamiento, {len(val_labels)} validación")
    
    # Crear modelo
    model = ImprovedSiameseNetwork(input_shape=(96, 96, 1), margin=1.5)
    
    logger.info(f"Modelo creado con {model.model.count_params():,} parámetros")
    
    # Callbacks optimizados - solo el mejor modelo
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy_metric',  # Monitorear precisión en lugar de pérdida
            patience=15,
            restore_best_weights=True,
            verbose=1,
            mode='max'  # Maximizar precisión
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy_metric',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
            mode='max'
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_fingerprint_model.h5',
            monitor='val_accuracy_metric',
            save_best_only=True,
            verbose=1,
            mode='max'
        )
    ]
    
    # Entrenar modelo
    logger.info("Iniciando entrenamiento...")
    history = model.model.fit(
        [train_a, train_b],
        train_labels,
        validation_data=([val_a, val_b], val_labels),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Visualizar historia de entrenamiento
    visualizer.plot_training_history(history)
    
    # Evaluar modelo
    logger.info("=== EVALUACIÓN FINAL ===")
    
    # Calcular predicciones en validación
    val_predictions = model.model.predict([val_a, val_b], verbose=0)
    threshold = model.margin / 2.0
    
    # Matriz de confusión
    visualizer.plot_confusion_matrix(val_labels, val_predictions.flatten(), threshold)
    
    # Análisis de distancias
    same_user_mask = val_labels == 1
    diff_user_mask = val_labels == 0
    
    distances_same = val_predictions.flatten()[same_user_mask]
    distances_diff = val_predictions.flatten()[diff_user_mask]
    
    visualizer.plot_distance_distribution(distances_same, distances_diff, threshold)
    
    # Métricas finales
    val_pred_binary = (val_predictions.flatten() < threshold).astype(int)
    val_accuracy = np.mean(val_pred_binary == val_labels)
    
    logger.info(f"Precisión en validación: {val_accuracy:.4f}")
    logger.info(f"Umbral usado: {threshold:.4f}")
    
    # Estadísticas de distancias
    logger.info(f"Distancia promedio mismo usuario: {np.mean(distances_same):.4f} ± {np.std(distances_same):.4f}")
    logger.info(f"Distancia promedio usuarios diferentes: {np.mean(distances_diff):.4f} ± {np.std(distances_diff):.4f}")
    
    # Análisis de embeddings
    sample_embeddings = model.get_embeddings(val_a[:20])
    embedding_norms = np.linalg.norm(sample_embeddings, axis=1)
    
    logger.info(f"Norma promedio de embeddings: {np.mean(embedding_norms):.4f}")
    logger.info(f"Rango de normas: [{np.min(embedding_norms):.4f}, {np.max(embedding_norms):.4f}]")
    
    # Solo informar sobre el mejor modelo guardado
    logger.info(f"✅ Mejor modelo guardado: best_fingerprint_model.h5")
    logger.info(f"📊 Mejor precisión en validación: {max(history.history['val_accuracy_metric']):.4f}")
    
    # Resumen final
    logger.info("\n" + "="*50)
    logger.info("RESUMEN DEL ENTRENAMIENTO")
    logger.info("="*50)
    logger.info(f"Usuarios en dataset: {len(set([img['user_idx'] for img in images_data]))}")
    logger.info(f"Imágenes totales: {len(images_data)}")
    logger.info(f"Pares para entrenamiento: {len(train_labels)}")
    logger.info(f"Pares para validación: {len(val_labels)}")
    logger.info(f"Épocas entrenadas: {len(history.history['loss'])}")
    logger.info(f"Mejor precisión validación: {max(history.history['val_accuracy_metric']):.4f}")
    logger.info(f"Precisión final: {val_accuracy:.4f}")
    logger.info("="*50)
    
    return model, history

if __name__ == "__main__":
    # Configurar TensorFlow
    tf.random.set_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # Configurar GPU si está disponible
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"GPU detectada: {len(gpus)} dispositivo(s)")
        except RuntimeError as e:
            logger.warning(f"Error configurando GPU: {e}")
    else:
        logger.info("Entrenando en CPU")
    
    # Entrenar modelo
    model, history = train_fingerprint_model()
