import cv2
import numpy as np

def preprocess_for_ocr(warped_image):
    # 1. Grisetara bihurtu (informazio kolorea kendu baina intentsitatea mantendu)
    gray = cv2.cvtColor(warped_image, cv2.COLOR_BGR2GRAY)
    
    # 2. CLAHE aplikatu: Kontrastea hobetu xehetasunak galdu gabe
    # clipLimit altuagoak kontraste gehiago ematen du, baina zarata igo dezake
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_gray = clahe.apply(gray)
    
    # 3. Bilateral Filter: Zarata kendu letrak lausotu gabe
    # (d=9, sigmaColor=75, sigmaSpace=75)
    processed = cv2.bilateralFilter(enhanced_gray, 9, 75, 75)
    
    # 4. Sharpening (Zorroztu): Letren ertzak markatuago egon daitezen
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(processed, -1, kernel)
    
    return sharpened

def order_points(pts):
    """Orders 4 points: [top-left, top-right, bottom-right, bottom-left]"""
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left has smallest sum, bottom-right has largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # Top-right has smallest difference, bottom-left has largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect



def get_perspective_transform(image, pts, margin_ratio=0.1):
    """Lau puntuko transformazioa marjina gehigarri batekin jatorrian."""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # 1. Kalkulatu dimentsioak (zure jatorrizko logikarekin)
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # 2. Kalkulatu desplazamenduak (offset) marjinaren arabera
    dx = maxWidth * margin_ratio
    dy = maxHeight * margin_ratio

    # 3. Zabaldu JATORRIZKO puntuak (rect) kanporantz
    # Kontuz: irudiaren mugetatik ez ateratzeko (opcional: np.clip erabili daiteke)
    rect_expanded = np.array([
        [tl[0] - dx, tl[1] - dy], # Top-left (gora eta ezkerrera)
        [tr[0] + dx, tr[1] - dy], # Top-right (gora eta eskuinera)
        [br[0] + dx, br[1] + dy], # Bottom-right (behera eta eskuinera)
        [bl[0] - dx, bl[1] + dy]  # Bottom-left (behera eta ezkerrera)
    ], dtype="float32")

    # 4. Helburuko puntuak (dst) berdin mantentzen dira, 
    # baina tamaina berria (maxWidth + 2*dx) izango da
    newWidth = int(maxWidth + 2 * dx)
    newHeight = int(maxHeight + 2 * dy)
    
    dst = np.array([
        [0, 0],
        [newWidth - 1, 0],
        [newWidth - 1, newHeight - 1],
        [0, newHeight - 1]], dtype="float32")
    
    # 5. Transformazioa aplikatu zabaldutako puntuekin
    M = cv2.getPerspectiveTransform(rect_expanded, dst)
    warped = cv2.warpPerspective(image, M, (newWidth, newHeight))
    
    return warped
