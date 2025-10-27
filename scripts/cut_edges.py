import cv2
import numpy as np
import os
from skimage.exposure import match_histograms


def get_image_paths(directory, exts=(".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(exts):
                image_paths.append(os.path.join(root, file))
    return sorted(image_paths)





def detect_ekg_contour(image_path, save_debug=True):
    # 1️⃣ Wczytaj obraz
    img = cv2.imread(image_path)
    original = img.copy()

    # 2️⃣ Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Wygładzenie i krawędzie (Canny)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 50, 150)

    # 4️⃣ Wypełnienie dziur, żeby kontur był spójny
    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    # 5️⃣ Znajdź wszystkie kontury
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ Nie znaleziono konturów.")
        return img

    # 6️⃣ Wybierz największy kontur (największy obszar)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 7️⃣ Narysuj prostokąt wokół kartki EKG
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # 8️⃣ Wytnij fragment EKG
    ekg_cropped = img[y:y+h, x:x+w]

    # 9️⃣ (Opcjonalny debug)
    if save_debug:
        cv2.imwrite("debug_edges.jpg", edges)
        cv2.imwrite("debug_ekg_box.jpg", original)
        cv2.imwrite("ekg_cropped_detected.jpg", ekg_cropped)
        print("✅ Zapisano pliki debugujące (debug_ekg_box.jpg, ekg_cropped_detected.jpg)")

    return ekg_cropped



import cv2
import numpy as np

def detect_ekg_contour_color_sensitive(image_path, save_debug=True):
    # 1️⃣ Wczytaj obraz
    img = cv2.imread(image_path)
    original = img.copy()

    # 2️⃣ Przejdź do HSV (łatwiej odróżnić barwy)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3️⃣ Wybierz zakres kolorów EKG (ciepłe odcienie: róż, czerwony, jasny beż)
    # Zakres HSV dopasowany do czerwonych linii EKG i jasnego papieru
    lower_warm = np.array([0, 20, 160])
    upper_warm = np.array([30, 200, 255])
    mask_warm = cv2.inRange(hsv, lower_warm, upper_warm)

    # 4️⃣ Dodatkowo jasny papier (żeby objąć całą kartkę)
    lower_white = np.array([0, 0, 150])
    upper_white = np.array([180, 60, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # 5️⃣ Połącz obie maski
    mask = cv2.bitwise_or(mask_warm, mask_white)

    # 6️⃣ Morfologia — zamknięcie dziur i usunięcie szumów
    kernel = np.ones((25,25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 7️⃣ Znajdź kontury
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Nie znaleziono konturów EKG (spróbuj zwiększyć jasność zdjęcia).")
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 8️⃣ Narysuj ramkę i wytnij EKG
    cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)
    ekg_cropped = img[y:y+h, x:x+w]

    # 9️⃣ (Opcjonalny debug)
    if save_debug:
        cv2.imwrite("debug_mask_hsv.jpg", mask)
        cv2.imwrite("debug_color_box.jpg", original)
        cv2.imwrite("ekg_cropped_color_detected.jpg", ekg_cropped)
        print("✅ Zapisano: debug_mask_hsv.jpg, debug_color_box.jpg, ekg_cropped_color_detected.jpg")

    return original, ekg_cropped

import cv2
import numpy as np

def cut_ekg_by_texture(image_path, save_debug=False):
    # Wczytanie obrazu
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1️⃣ Wyrównanie oświetlenia
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray_eq = cv2.equalizeHist(gray)

    # 2️⃣ Obliczenie lokalnej wariancji (tekstury kratki)
    kernel = np.ones((15,15), np.float32) / 225
    mean = cv2.filter2D(gray_eq, -1, kernel)
    sqr_mean = cv2.filter2D(gray_eq**2, -1, kernel)
    variance = sqr_mean - mean**2
    texture = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 3️⃣ Segmentacja: tylko mocno teksturowane obszary
    _, mask = cv2.threshold(texture, 20, 255, cv2.THRESH_BINARY)

    # 4️⃣ Czyszczenie maski (usuwa drobne błędy i cienie)
    kernel2 = np.ones((35,35), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    # 5️⃣ Znajdź największy obszar = kratka EKG
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Nie znaleziono kratki EKG.")
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 6️⃣ Wycięcie obszaru EKG
    cropped = img[y:y+h, x:x+w]

    # 7️⃣ (Opcjonalnie) zapis debugowy
    if save_debug:
        dbg = img.copy()
        cv2.drawContours(dbg, [largest], -1, (0,255,0), 3)
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (255,0,0), 3)
        cv2.imwrite("debug_texture_mask.jpg", mask)
        cv2.imwrite("debug_texture_detected.jpg", dbg)

    cv2.imwrite("ekg_cropped_by_texture.jpg", cropped)
    print("✅ Zapisano: ekg_cropped_by_texture.jpg")
    return cropped


import cv2
import numpy as np
from skimage.exposure import match_histograms

def cut_edges_hist_texture(target_path, reference_path, save_debug=False):
    img_ref = cv2.imread(reference_path)
    img = cv2.imread(target_path)
    H, W = img.shape[:2]

    # 1️⃣ Dopasowanie histogramu do wzorca
    matched = match_histograms(img, img_ref, channel_axis=-1)

    # 2️⃣ Konwersja na odcienie szarości
    gray = cv2.cvtColor(matched, cv2.COLOR_BGR2GRAY)

    # 3️⃣ Wyrównanie oświetlenia
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    gray = cv2.equalizeHist(gray)

    # 4️⃣ Oblicz lokalną wariancję (teksturę kratki)
    kernel = np.ones((15,15), np.float32) / 225
    mean = cv2.filter2D(gray, -1, kernel)
    sqr_mean = cv2.filter2D(gray**2, -1, kernel)
    variance = sqr_mean - mean**2
    texture = cv2.normalize(variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 5️⃣ Zrób maskę tylko z obszarów o dużej teksturze
    _, mask = cv2.threshold(texture, 20, 255, cv2.THRESH_BINARY)

    # 6️⃣ Morfologia, żeby oczyścić maskę
    kernel2 = np.ones((25,25), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel2)

    # 7️⃣ Znajdź największy obszar (to powinna być kartka EKG)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ Nie znaleziono kratki EKG.")
        return img

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # 8️⃣ Wytnij wykryty obszar
    cropped = img[y:y+h, x:x+w]

    # 9️⃣ (Opcjonalny debug)
    if save_debug:
        dbg = img.copy()
        cv2.drawContours(dbg, [largest], -1, (0,255,0), 3)
        cv2.rectangle(dbg, (x, y), (x+w, y+h), (255,0,0), 2)
        cv2.imwrite("debug_mask_texture.jpg", mask)
        cv2.imwrite("debug_detected_area.jpg", dbg)

    cv2.imwrite("ekg_hist_texture_cropped.jpg", cropped)
    print("✅ Zapisano: ekg_hist_texture_cropped.jpg")
    return cropped


def main():
    directory = "./Data/10140238"
    image_paths = get_image_paths(directory)

    for path in image_paths:
        cropped_image = detect_ekg_contour(path)
        if cropped_image is not None:
            save_path = os.path.join("contur", os.path.basename(path))
            os.makedirs("contur", exist_ok=True)
            cv2.imwrite(save_path, cropped_image)
            print(f"Zapisano przycięty obraz do: {save_path}")
    print("Przycinanie zakończone.")

    #cropped_image = cut_ekg_by_texture(image_paths[3])
    #cv2.imwrite("dtexture_cut.jpg", cropped_image)
    #cv2.imshow("Przycięty EKG", cropped_image)

    #cropped_image = cut_edges_hist_texture(image_paths[3], image_paths[0], save_debug=False)
    #cv2.imwrite("hist_cut.jpg", cropped_image)

 
    # Wczytaj obraz w trybie grayscale (lub zmień na kolorowy, jeśli potrzebujesz)
    #img = cv2.imread(image_paths[3], cv2.IMREAD_GRAYSCALE)

# Tworzymy obiekt CLAHE
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

# Zastosowanie CLAHE
    #cl1 = clahe.apply(img)

# Zapisz wynik
    #cv2.imwrite("clahe_output.jpg", cl1)
    #cropped_image = cut_ekg_by_texture("clahe_output.jpg")
    #cv2.imwrite("popraw kontrast.jpg", cropped_image)


    #alpha = 1.5
    #beta = 20
    #img = cv2.imread(image_paths[3])
    #new_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #cv2.imwrite("alfabetacontrast.jpg", new_img)
    #cropped_image = cut_ekg_by_texture("alfabetacontrast.jpg")
    #cv2.imwrite("popraw kontrast.jpg", cropped_image)


    #cropped_image = detect_ekg_contour_color_sensitive("alfabetacontrast.jpg")
    #cv2.imwrite("color_sensitive_cut.jpg", cropped_image)


if __name__ == "__main__":
    main()