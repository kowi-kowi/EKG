from PIL import Image, ImageEnhance
import cv2
import numpy as np
import os


def detect_ekg_contour(image_path):
    # 1 Wczytaj obraz
    img = cv2.imread(image_path)
    original = img.copy()

    # 2 popraw kontrast jasnosc ciemnosc
    alpha = 1.5
    beta = 20
    img_con = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    #img = cv2.convertScaleAbs(img, alpha=1.0, beta=-70)
    #jasny, jasny z kontrastem, ciemny, ciemny z kontrastem
    possible_images = [img,
                       img_con,
                       cv2.convertScaleAbs(img, alpha=1.0, beta=-70),
                       cv2.convertScaleAbs(img, alpha=1.5, beta=50),
                       cv2.convertScaleAbs(img_con, alpha=1.0, beta=-70),
                       cv2.convertScaleAbs(img_con, alpha=1.5, beta=50)
                       ]
    for image in possible_images:
        
    # 3 Konwersja do skali szarości
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

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
            return image

    # 7 Wybierz największy kontur (największy obszar)
        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

    # 8 Narysuj prostokąt wokół kartki EKG
        cv2.rectangle(original, (x, y), (x+w, y+h), (0, 255, 0), 3)

    # 9 Wytnij fragment EKG
        ekg_cropped = img[y:y+h, x:x+w]

        if image.shape[:2] == ekg_cropped.shape[:2]:
            pass
        else:
            return ekg_cropped

    print("Nie udało się wykryć konturu na żadnym z wariantów przetwarzania obrazu.")
    return original

ob1 = '/Users/maria/projekty/EKG/cropped1/102150619-0010.png'
ob2 = '/Users/maria/projekty/EKG/cropped1/102150619-0009.png'
ob3 = '/Users/maria/projekty/EKG/cropped1/102150619-0005.png'
img = Image.open(ob3)


ciemny_crop = detect_ekg_contour(ob1)
cv2.imwrite("obraz.jpg", ciemny_crop)


