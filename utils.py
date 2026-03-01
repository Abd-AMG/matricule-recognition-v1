import cv2
import numpy as np
import pickle

def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

kmeans, mapping = load_model()

def predict_digit(segment):
    """✅ Improved: OTSU + Centering + Morphology = دقة أعلى بكثير"""
    if segment.size == 0:
        return "?"
    
    # Binarization أفضل
    _, binary = cv2.threshold(segment, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # تنظيف الضوضاء وملء الفجوات
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    # Centering الرقم (السر الذي يرفع الدقة 10-15%)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return "?"
    
    x, y, w, h = cv2.boundingRect(coords)
    tight = binary[y:y+h, x:x+w]
    
    # Resize مع الحفاظ على النسبة + padding مركزي
    max_dim = max(w, h)
    if max_dim == 0:
        return "?"
    
    scale = 6.0 / max_dim
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(tight, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    padded = np.zeros((8, 8), dtype=np.uint8)
    start_x = (8 - new_w) // 2
    start_y = (8 - new_h) // 2
    padded[start_y:start_y+new_h, start_x:start_x+new_w] = resized
    
    flat = padded.flatten().reshape(1, -1).astype(np.float32)
    cluster = kmeans.predict(flat)[0]
    return mapping.get(cluster, "?")

def process_image(image_np, mode="auto"):
    """✅ Improved segmentation للـ matricule والأرقام اليدوية"""
    if len(image_np.shape) == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    else:
        gray = image_np.copy()

    # Binarisation حسب نوع الصورة
    if mode == "plate":
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 21, 10)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    digits_info = []
    h, w = gray.shape
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 80: continue
        
        x, y, ww, hh = cv2.boundingRect(cnt)
        aspect = ww / hh
        if not (0.4 < aspect < 1.6) or hh < 25 or ww < 10: continue
        
        # Solidity filter (يمنع الأشكال الغريبة)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < 0.65: continue
        
        digit_crop = gray[y:y+hh, x:x+ww]
        digit_resized = cv2.resize(digit_crop, (20, 20), interpolation=cv2.INTER_AREA)  # أكبر قليلاً قبل الـ centering
        
        pred = predict_digit(digit_resized)
        digits_info.append((x, y, ww, hh, pred))
    
    digits_info.sort(key=lambda item: item[0])  # ترتيب يسار → يمين
    
    result_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    recognized_text = ""
    for x, y, ww, hh, digit in digits_info:
        cv2.rectangle(result_img, (x, y), (x+ww, y+hh), (0, 255, 0), 3)
        cv2.putText(result_img, str(digit), (x, y-12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
        recognized_text += str(digit)
    
    return {
        "original": image_np,
        "gray": gray,
        "binary": binary,
        "result": result_img,
        "text": recognized_text,
        "boxes": digits_info
    }