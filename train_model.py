"""
Archivo principal para entrenar el modelo de huellas dactilares
Punto de entrada simplificado del sistema
"""
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from training.train_fingerprint_model import train_fingerprint_model

def main():
    """
    Función principal para ejecutar el entrenamiento
    """
    print("=== SISTEMA DE RECONOCIMIENTO DE HUELLAS DACTILARES ===")
    print("Iniciando entrenamiento del modelo siamesa...")
    print()
    
    try:
        # Entrenar el modelo
        model, history = train_fingerprint_model()
        
        if model is not None:
            print("\n✅ Entrenamiento completado exitosamente!")
            print("📁 Modelo guardado:")
            print("   - best_fingerprint_model.h5 (mejor modelo con mayor precisión)")
            print("📊 Gráficas mostradas en pantalla durante el entrenamiento")
        else:
            print("\n❌ Error durante el entrenamiento")
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
