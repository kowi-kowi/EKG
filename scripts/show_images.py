import argparse
import glob
import math
import os
import random
from PIL import Image
import matplotlib.pyplot as plt

def find_images(directory, exts=("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")):
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(directory, e)))
    return sorted(files)

def show_images(paths, cols=3, figsize=(12, 8)):
    if not paths:
        print("Brak obrazów do wyświetlenia.")
        return
    rows = math.ceil(len(paths) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax in axes[len(paths):]:
        ax.axis("off")
    for ax, p in zip(axes, paths):
        img = Image.open(p)
        ax.imshow(img)
        ax.set_title(os.path.basename(p))
        ax.axis("off")
    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Wyświetl kilka obrazów z katalogu")
    parser.add_argument("--dir", default="images", help="katalog z obrazami")
    parser.add_argument("--num", type=int, default=6, help="liczba obrazów do pokazania")
    parser.add_argument("--cols", type=int, default=3, help="liczba kolumn w siatce")
    parser.add_argument("--random", action="store_true", help="losowy wybór obrazów")
    args = parser.parse_args()

    imgs = find_images(args.dir)
    if not imgs:
        print(f"Nie znaleziono obrazów w {args.dir}")
        return

    if args.random:
        imgs = random.sample(imgs, min(args.num, len(imgs)))
    else:
        imgs = imgs[: args.num]

    show_images(imgs, cols=args.cols)

if __name__ == "__main__":
    main()