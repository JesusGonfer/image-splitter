import cv2
import numpy as np
import os
import sys

def merge_contours(contours, min_distance=50):
    """
    Fusiona contornos si están muy cerca unos de otros.
    :param contours: Lista de contornos (rectángulos delimitadores).
    :param min_distance: Distancia mínima para fusionar contornos.
    :return: Lista de contornos fusionados.
    """
    merged = []
    contours = sorted(contours, key=lambda b: b[1])  # Ordenar contornos por la posicion vertical (de arriba hacia abajo)

    while contours:
        contour = contours.pop(0)
        x, y, w, h = contour
        merged_contour = (x, y, w, h)

        to_merge = [contour]
        i = 0
        while i < len(contours):
            ox, oy, ow, oh = contours[i]
            if abs(x - ox) < min_distance and abs(y - oy) < min_distance:
                to_merge.append(contours.pop(i))
                merged_contour = (min(merged_contour[0], ox), min(merged_contour[1], oy),
                                  max(merged_contour[0] + merged_contour[2], ox + ow) - merged_contour[0],
                                  max(merged_contour[1] + merged_contour[3], oy + oh) - merged_contour[1])
            else:
                i += 1

        merged.append(merged_contour)

    return merged

def split_images_by_borders(input_image_path, output_dir, margin=0, min_distance=50):

    image = cv2.imread(input_image_path)
    if image is None:
        print(f"Error: No se pudo cargar la imagen {input_image_path}.")
        return

    margin_left = 50  # Ajusta este valor si deseas ignorar un margen del escaner
    image = image[:, margin_left:]


    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gamma = 2.0
    look_up_table = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)], dtype=np.uint8)
    normalized_gray = cv2.LUT(gray, look_up_table)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(normalized_gray)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 

    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 9, 2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    os.makedirs(output_dir, exist_ok=True)

    debug_dir = os.path.join(output_dir, 'debug')
    os.makedirs(debug_dir, exist_ok=True)

    image_area = image.shape[0] * image.shape[1]
    min_area = 0.05 * image_area  # Ajusta el porcentaje a un valor menor para capturar objetos pequeños

    valid_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w * h > min_area:  
            valid_contours.append((x, y, w, h))

    valid_contours = merge_contours(valid_contours, min_distance)

    valid_contours = sorted(valid_contours, key=lambda b: b[1])

    base_name = os.path.splitext(os.path.basename(input_image_path))[0]

    debug_image = image.copy()
    for x, y, w, h in valid_contours:
        x_adj = max(0, x - margin)
        y_adj = max(0, y - margin)
        w_adj = min(image.shape[1] - x_adj, w + 2 * margin)
        h_adj = min(image.shape[0] - y_adj, h + 2 * margin)

        cv2.rectangle(debug_image, (x_adj, y_adj), (x_adj + w_adj, y_adj + h_adj), (0, 255, 0), 2)

    debug_path = os.path.join(debug_dir, f"{base_name}_debug.jpg")
    cv2.imwrite(debug_path, debug_image)

    count = 1
    for x, y, w, h in valid_contours:
        x_adj = max(0, x - margin)
        y_adj = max(0, y - margin)
        w_adj = min(image.shape[1] - x_adj, w + 2 * margin)
        h_adj = min(image.shape[0] - y_adj, h + 2 * margin)

        sub_image = image[y_adj:y_adj+h_adj, x_adj:x_adj+w_adj]
        output_path = os.path.join(output_dir, f"{base_name}_{count:03d}.jpg")
        cv2.imwrite(output_path, sub_image)
        count += 1

    print(f"Se guardaron {count-1} subimágenes en el directorio '{output_dir}'.")
    print(f"Imagen de depuración guardada en '{debug_path}'.")

def process_folder(input_folder_path, output_dir, margin=0, min_distance=50):
    if not os.path.isdir(input_folder_path):
        print(f"Error: La ruta proporcionada no es una carpeta válida.")
        return

    for filename in os.listdir(input_folder_path):
        input_image_path = os.path.join(input_folder_path, filename)
        if os.path.isfile(input_image_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            print(f"Procesando la imagen: {input_image_path}")
            split_images_by_borders(input_image_path, output_dir, margin, min_distance)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python script.py <ruta_imagen_o_carpeta> <directorio_salida> <margen>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]
    margin = int(sys.argv[3]) if len(sys.argv) > 3 else 0  
    min_distance = int(sys.argv[4]) if len(sys.argv) > 4 else 50  

    if os.path.isdir(input_path):
        process_folder(input_path, output_dir, margin, min_distance)
    else:
        split_images_by_borders(input_path, output_dir, margin, min_distance)
