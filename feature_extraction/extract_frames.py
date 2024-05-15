"""
Script to extract frames

Usage: 
    extract_frames.py --videos_dir=<videos_dir> [options]

Options:
    -h --help                             Show this screen.

Arguments:
    <videos_dir>                          Path to the folder containing the videos.
"""

import cv2
import os
from docopt import docopt
from tqdm import tqdm

def extract_frames(video_path, output_folder):
    """
    Extrae frames de un video y los guarda en una carpeta específica.
    :param video_path: Ruta completa al archivo de video.
    :param output_folder: Carpeta donde se guardarán los frames.
    """
    # Asegurar que la carpeta de salida existe
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Abrir el video
    cap = cv2.VideoCapture(video_path)
    count = 0
    
    # Leer frame por frame
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Si no hay más frames, salir del bucle
        
        # Guardar frame como imagen
        resized_frame = cv2.resize(frame, (224, 224))
        cv2.imwrite(os.path.join(output_folder, f"frame_{count:04d}.jpg"), resized_frame)
        count += 1

    # Liberar el objeto cap y cerrar todas las ventanas abiertas
    cap.release()

def extract_frames_from_folder(folder_path):
    """
    Busca todos los archivos de video en la carpeta y subcarpetas y extrae sus frames.
    :param folder_path: Ruta a la carpeta donde se buscan los videos.
    """
    # Recorrer la carpeta y subcarpetas
    for root, dirs, files in os.walk(folder_path):
        for file in tqdm(files):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Añade más formatos de video si es necesario
                video_path = os.path.join(root, file)
                # Crear una carpeta específica para los frames de este video
                output_folder = os.path.join(folder_path, 'frames', file.split('.')[0])
                # Extraer frames
                extract_frames(video_path, output_folder)
                #print(f"Frames extraídos para {video_path} en {output_folder}")

# Uso del script
if __name__ == "__main__":
    args = docopt(__doc__)
    root_folder = args['--videos_dir']
    extract_frames_from_folder(root_folder)
