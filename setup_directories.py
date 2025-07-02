#!/usr/bin/env python3
"""
Script para crear los directorios necesarios del proyecto
"""
import os

def create_directories():
    """Crea los directorios necesarios para el proyecto"""
    directories = [
        "models/saved",
        "data/embeddings",
        "logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Directorio creado/verificado: {directory}")
    
    print("🎉 Configuración de directorios completada")

if __name__ == "__main__":
    create_directories()
