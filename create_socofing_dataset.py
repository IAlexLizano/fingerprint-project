"""
Script para crear el dataset JSON a partir de las imágenes SOCOFing Altered-Easy
"""

import os
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from collections import defaultdict

def parse_filename(filename: str) -> Tuple[str, str, str, str, str]:
    """
    Parsea el nombre del archivo para extraer información
    Formato: {person_id}__{gender}_{hand}_{finger}_{alteration}.BMP
    
    Returns:
        (person_id, gender, hand, finger, alteration)
    """
    try:
        # Remover extensión
        name = filename.replace('.BMP', '')
        
        # Dividir por '__'
        parts = name.split('__')
        if len(parts) != 2:
            return None
        
        person_id = parts[0]
        rest = parts[1]
        
        # Dividir el resto por '_'
        info_parts = rest.split('_')
        if len(info_parts) < 4:
            return None
        
        gender = info_parts[0]
        hand = info_parts[1]
        
        # El dedo puede tener múltiples partes (ej: middle_finger)
        # La alteración es siempre la última parte
        alteration = info_parts[-1]
        finger_parts = info_parts[2:-1]
        finger = '_'.join(finger_parts)
        
        return person_id, gender, hand, finger, alteration
    except:
        return None

def load_image_info(image_path: str) -> Dict:
    """Carga información básica de una imagen"""
    try:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        
        height, width = img.shape
        return {
            "path": image_path,
            "width": width,
            "height": height,
            "channels": 1  # Grayscale
        }
    except Exception as e:
        print(f"Error cargando imagen {image_path}: {e}")
        return None

def create_socofing_dataset(source_dir: str, output_file: str):
    """
    Crea el dataset JSON a partir de las imágenes SOCOFing
    
    Args:
        source_dir: Directorio con las imágenes SOCOFing
        output_file: Archivo JSON de salida
    """
    print(f"Procesando imágenes desde: {source_dir}")
    
    # Estructura del dataset
    dataset = {
        "name": "SOCOFing Altered-Easy",
        "description": "Dataset de huellas dactilares SOCOFing con alteraciones",
        "version": "1.0",
        "total_subjects": 0,
        "total_images": 0,
        "image_format": "BMP",
        "subjects": {}
    }
    
    # Agrupar imágenes por persona
    subjects_data = defaultdict(lambda: {
        "gender": "",
        "images": []
    })
    
    # Procesar todas las imágenes
    image_files = [f for f in os.listdir(source_dir) if f.endswith('.BMP')]
    print(f"Encontradas {len(image_files)} imágenes")
    
    processed_count = 0
    for filename in image_files:
        # Parsear nombre del archivo
        parsed = parse_filename(filename)
        if not parsed:
            print(f"No se pudo parsear: {filename}")
            continue
        
        person_id, gender, hand, finger, alteration = parsed
        
        # Ruta completa de la imagen
        image_path = os.path.join(source_dir, filename)
        
        # Cargar información de la imagen
        img_info = load_image_info(image_path)
        if not img_info:
            continue
        
        # Crear entrada para la imagen
        image_entry = {
            "filename": filename,
            "path": image_path,
            "hand": hand,
            "finger": finger,
            "alteration": alteration,
            "width": img_info["width"],
            "height": img_info["height"],
            "channels": img_info["channels"]
        }
        
        # Agregar a la estructura de datos
        subjects_data[person_id]["gender"] = gender
        subjects_data[person_id]["images"].append(image_entry)
        
        processed_count += 1
        if processed_count % 100 == 0:
            print(f"Procesadas {processed_count} imágenes...")
    
    # Convertir a formato final
    for person_id, data in subjects_data.items():
        dataset["subjects"][person_id] = {
            "subject_id": person_id,
            "gender": data["gender"],
            "total_images": len(data["images"]),
            "images": data["images"]
        }
    
    # Estadísticas finales
    dataset["total_subjects"] = len(dataset["subjects"])
    dataset["total_images"] = sum(len(s["images"]) for s in dataset["subjects"].values())
    
    # Guardar dataset
    with open(output_file, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"\nDataset creado exitosamente:")
    print(f"- Total de sujetos: {dataset['total_subjects']}")
    print(f"- Total de imágenes: {dataset['total_images']}")
    print(f"- Archivo guardado en: {output_file}")
    
    # Mostrar algunos ejemplos
    print(f"\nEjemplos de sujetos:")
    for i, (subject_id, subject_data) in enumerate(list(dataset["subjects"].items())[:5]):
        print(f"  Sujeto {subject_id}: {subject_data['gender']}, {subject_data['total_images']} imágenes")
        if subject_data["images"]:
            print(f"    Ejemplo: {subject_data['images'][0]['filename']}")
    
    return dataset

def analyze_dataset_distribution(dataset: Dict):
    """Analiza la distribución del dataset"""
    print("\n=== ANÁLISIS DEL DATASET ===")
    
    # Distribución por género
    gender_count = defaultdict(int)
    for subject_data in dataset["subjects"].values():
        gender_count[subject_data["gender"]] += 1
    
    print(f"Distribución por género:")
    for gender, count in gender_count.items():
        print(f"  {gender}: {count} sujetos")
    
    # Distribución por dedos
    finger_count = defaultdict(int)
    alteration_count = defaultdict(int)
    hand_count = defaultdict(int)
    
    for subject_data in dataset["subjects"].values():
        for image in subject_data["images"]:
            finger_count[image["finger"]] += 1
            alteration_count[image["alteration"]] += 1
            hand_count[image["hand"]] += 1
    
    print(f"\nDistribución por dedo:")
    for finger, count in sorted(finger_count.items()):
        print(f"  {finger}: {count} imágenes")
    
    print(f"\nDistribución por alteración:")
    for alteration, count in sorted(alteration_count.items()):
        print(f"  {alteration}: {count} imágenes")
    
    print(f"\nDistribución por mano:")
    for hand, count in sorted(hand_count.items()):
        print(f"  {hand}: {count} imágenes")

if __name__ == "__main__":
    # Configuración
    source_directory = r"C:\development\septimo\ia\projects\final_project\project-fingerprint\data\Altered-Easy"
    output_file = r"C:\development\septimo\ia\projects\final_project\project-fingerprint\data\socofing_dataset.json"
    
    # Verificar que el directorio existe
    if not os.path.exists(source_directory):
        print(f"Error: No se encuentra el directorio {source_directory}")
        exit(1)
    
    # Crear dataset
    dataset = create_socofing_dataset(source_directory, output_file)
    
    # Analizar distribución
    analyze_dataset_distribution(dataset)
    
    print(f"\n¡Dataset SOCOFing creado exitosamente!")
    print(f"Ahora puedes usar el archivo {output_file} para entrenar el modelo.")
