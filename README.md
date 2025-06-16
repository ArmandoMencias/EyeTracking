# Eye Typing System

Un sistema de teclado virtual controlado con la mirada, que permite escribir texto y recibir sugerencias de palabras en tiempo real. Utiliza MediaPipe para el seguimiento del iris, OpenCV para la interfaz gráfica, y pyttsx3 para la síntesis de voz.

## Contenido del repositorio

corpus.txt          #Texto base para el predictor de palabras
palabras.txt        #Lista de palabras para sugerencias adicionales
teclado.py          #Script principal
requirements.txt    #Dependencias de Python

## Prerrequisitos

- **Python 3.8+**  
- Pip (gestor de paquetes de Python)  
- Cámara web funcional  

## Instalación

1. **Clonar el repositorio**  
   ```bash
   git clone https://github.com/tu_usuario/tu_repositorio.git
   cd tu_repositorio

2. Crear y activar un entorno virtual
    python -m venv EyeTracking
    # Windows
    EyeTracking\Scripts\activate
    # macOS/Linux
    source EyeTracking/bin/activate

3. ### Instalar dependencias
    pip install --upgrade pip
    pip install -r requirements.txt

4. Ejecutar el teclado
    python teclado.py


# Flujo de escritura

Calibración: Sigue los puntos rojos en pantalla. Mantén la mirada fija 1s sobre cada punto.

Dwell sobre teclas: Posiciona el cursor verde sobre una tecla 1s para “pulsarla”.

Sugerencias: Tras escribir una palabra, aparecerán hasta 3 sugerencias basadas en tu corpus; mantén la mirada 1 s para seleccionarlas.

Voz: Cada carácter (y “eñe” / “arroba”) se pronuncia en español al seleccionarlo.

# Salir

Pulsa Esc o cierra la ventana para finalizar la aplicación.
