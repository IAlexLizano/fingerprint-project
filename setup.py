#!/usr/bin/env python3
"""
Script de configuración inicial para el Sistema de Autenticación de Huellas Dactilares
"""

import os
import sys
import subprocess
import shutil

def create_directories():
    """Crear directorios necesarios para el proyecto"""
    directories = [
        "models/saved",
        "data/dataset", 
        "data/embeddings",
        "data/datasets",
        "logs"
    ]
    
    print("Creando directorios...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    print("\nVerificando dependencias...")
    
    # Mapeo de nombres de paquetes a nombres de módulos
    package_mapping = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn", 
        "tensorflow": "tensorflow",
        "opencv-python": "cv2",
        "numpy": "numpy",
        "scikit-learn": "sklearn",
        "pydantic": "pydantic",
        "Pillow": "PIL"
    }
    
    missing_packages = []
    
    for package, module in package_mapping.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (faltante)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Faltan las siguientes dependencias: {', '.join(missing_packages)}")
        print("Ejecuta: pip install -r requirements.txt")
        return False
    
    print("✅ Todas las dependencias están instaladas")
    return True

def generate_sample_dataset():
    """Generar un dataset de muestra para pruebas"""
    print("\nGenerando dataset de muestra...")
    
    try:
        from utils.dataset_generator import DatasetGenerator
        
        generator = DatasetGenerator()
        dataset_path = generator.create_sample_dataset()
        print(f"✅ Dataset de muestra creado en: {dataset_path}")
        return True
    except Exception as e:
        print(f"❌ Error al generar dataset: {e}")
        return False

def test_imports():
    """Probar que todos los módulos se pueden importar correctamente"""
    print("\nProbando imports...")
    
    modules = [
        "models.siamese_network",
        "data.dataset_manager", 
        "training.trainer",
        "services.fingerprint_service",
        "api.schemas",
        "utils.dataset_generator"
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            return False
    
    print("✅ Todos los módulos se importan correctamente")
    return True

def create_config_file():
    """Crear archivo de configuración"""
    config_content = """# Configuración del Sistema de Autenticación de Huellas Dactilares

# Configuración del modelo
MODEL_CONFIG = {
    "input_shape": (128, 128, 1),
    "embedding_dim": 128,
    "model_path": "models/saved/siamese_model.h5"
}

# Configuración del dataset
DATASET_CONFIG = {
    "dataset_path": "data/dataset",
    "embeddings_path": "data/embeddings",
    "embeddings_file": "data/embeddings/user_embeddings.npy",
    "users_file": "data/embeddings/users.json"
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "default_epochs": 50,
    "default_batch_size": 32,
    "default_validation_split": 0.2,
    "learning_rate": 0.0001
}

# Configuración de autenticación
AUTH_CONFIG = {
    "default_threshold": 0.5,
    "max_images_per_user": 10
}

# Configuración del servidor
SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "reload": True
}
"""
    
    with open("config.py", "w") as f:
        f.write(config_content)
    
    print("✅ Archivo de configuración creado: config.py")

def main():
    """Función principal de configuración"""
    print("=== Configuración del Sistema de Autenticación de Huellas Dactilares ===\n")
    
    # Crear directorios
    create_directories()
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Probar imports
    if not test_imports():
        print("❌ Error en los imports. Verifica la instalación.")
        sys.exit(1)
    
    # Generar dataset de muestra
    generate_sample_dataset()
    
    # Crear archivo de configuración
    create_config_file()
    
    print("\n=== Configuración completada ===")
    print("\nPara ejecutar el servidor:")
    print("  python main.py")
    print("\nPara ver la documentación:")
    print("  http://localhost:8000/docs")
    print("\nPara ejecutar el ejemplo:")
    print("  python examples/example_usage.py")

if __name__ == "__main__":
    main() 