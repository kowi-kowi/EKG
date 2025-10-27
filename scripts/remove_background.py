import cv2
import numpy as np
import os


def get_image_paths(directory, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)


def detect_ekg_contour(image_path):
    # 1 Wczytaj obraz
    img = cv2.imread(image_path)
    original = img.copy()

    # 2 popraw kontrast
    alpha = 1.5
    beta = 20
    #img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # 3 Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 4 Wygładzenie i krawędzie (Canny)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 5 Wypełnienie dziur, żeby kontur był spójny
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 6 Znajdź wszystkie kontury
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(" X Nie znaleziono konturów. X")
        return img

    # 7 Wybierz największy kontur (największy obszar)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 8 Narysuj prostokąt wokół kartki EKG
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # 9 Wytnij fragment EKG
    ekg_cropped = img[y:y+h, x:x+w]


    return ekg_cropped

def main():
    directory = "./Data"
    list_of_dirs = os.listdir(directory)
    print(list_of_dirs)
    image_paths = []

    for dir_name in list_of_dirs:
        imp = get_image_paths(directory + '/' + dir_name)
        image_paths = image_paths + imp

    print(image_paths)
    for path in image_paths:
        cropped_image = detect_ekg_contour(path)
        if cropped_image is not None:
            save_path = os.path.join("cropped1", os.path.basename(path))
            os.makedirs("cropped1", exist_ok=True)
            cv2.imwrite(save_path, cropped_image)
            print(f"Zapisano przycięty obraz do: {save_path}")
    print("Przycinanie zakończone.")

if __name__ == "__main__":
    main()