import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title='🚨 Highway Radar ANPR \nBy Abdelaziz AMGHOUGH',
    page_icon='🚗',
    layout='wide',
    initial_sidebar_state='expanded'
)

st.title('🚨 Highway Radar – Automatic Number Plate Recognition (ANPR)\nBy Abdelaziz AMGHOUGH')
st.markdown('''
**Professional-grade computer-vision pipeline** used in real highway radar systems.  
Combines everything from your class: grayscale + histograms, bilateral + Canny + contour plate detection, interactive binarization, **plus** K-means segmentation and extra powerful tools.
''')

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header('📤 Upload & Settings')
    uploaded_file = st.file_uploader('Upload vehicle image (JPEG/PNG)', type=['jpg', 'jpeg', 'png'])
    
    st.divider()
    st.subheader('Pipeline Controls')
    run_full = st.button('🚀 Run Full Radar Pipeline (Default Params)', type='primary', use_container_width=True)
    
    st.divider()
    st.caption('Made for educational & demonstration purposes – Marrakesh 2026')

# ====================== SESSION STATE ======================
if 'plate_cropped' not in st.session_state:
    st.session_state.plate_cropped = None
if 'img_with_contour' not in st.session_state:
    st.session_state.img_with_contour = None

# ====================== LOAD IMAGE ======================
if uploaded_file is None:
    st.info('👆 Upload a photo of a vehicle to start the radar pipeline')
    st.stop()

bytes_data = uploaded_file.getvalue()
np_arr = np.frombuffer(bytes_data, np.uint8)
img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ====================== TABS ======================
tab_overview, tab_preproc, tab_detection, tab_binarize, tab_kmeans, tab_pipeline = st.tabs([
    '📸 Overview', '📊 Preprocessing', '🔍 Plate Detection', '⚙️ Binarization', '🧠 K-Means', '🚀 Full Pipeline'
])

# ====================== TAB 1 – OVERVIEW ======================
with tab_overview:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Original Image')
        st.image(img_rgb, use_column_width=True)
    with col2:
        st.subheader('Grayscale')
        st.image(gray, use_column_width=True, channels='GRAY')
    
    st.subheader('Histogram (Image 1 style from class)')
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(cv2.calcHist([gray], [0], None, [256], [0, 256]), color='white')
    ax.set_xlim(0, 256)
    ax.set_facecolor('#0e1117')
    ax.grid(True, alpha=0.3)
    st.pyplot(fig)

# ====================== TAB 2 – PREPROCESSING ======================
with tab_preproc:
    st.subheader('Noise Reduction (Bilateral Filter – exactly as in your class)')
    d = st.slider('Diameter', 5, 25, 11)
    sc = st.slider('Sigma Color', 10, 100, 17)
    ss = st.slider('Sigma Space', 10, 100, 17)
    
    blur = cv2.bilateralFilter(gray, d, sc, ss)
    st.image(blur, caption='Bilateral Filtered', use_column_width=True, channels='GRAY')

# ====================== TAB 3 – PLATE DETECTION ======================
with tab_detection:
    st.subheader('License Plate Detection (Full contour algorithm from class)')
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        bil_d = st.slider('Bilateral Diameter', 5, 25, 11, key='bil_d')
        bil_sc = st.slider('Sigma Color', 10, 100, 17, key='bil_sc')
    with col_b:
        bil_ss = st.slider('Sigma Space', 10, 100, 17, key='bil_ss')
        canny_low = st.slider('Canny Low', 10, 100, 30, key='c_low')
    with col_c:
        canny_high = st.slider('Canny High', 100, 300, 200, key='c_high')
        approx_f = st.slider('Approx Factor', 0.005, 0.05, 0.018, step=0.001, key='approx_f')

    if st.button('🔍 Detect License Plate', type='primary', use_container_width=True):
        with st.spinner('Running radar detection...'):
            blur = cv2.bilateralFilter(gray, bil_d, bil_sc, bil_ss)
            edged = cv2.Canny(blur, canny_low, canny_high)
            
            contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
            
            plate_cnt = None
            for c in contours:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, approx_f * peri, True)
                if len(approx) == 4:
                    plate_cnt = approx
                    break
            
            if plate_cnt is not None:
                mask = np.zeros(gray.shape, dtype=np.uint8)
                cv2.drawContours(mask, [plate_cnt], -1, 255, -1)
                plate = cv2.bitwise_and(gray, gray, mask=mask)
                
                x, y, w, h = cv2.boundingRect(plate_cnt)
                plate_cropped = plate[y:y+h, x:x+w]
                
                # Save to session
                st.session_state.plate_cropped = plate_cropped
                
                # Draw on original
                img_draw = img_bgr.copy()
                cv2.drawContours(img_draw, [plate_cnt], -1, (0, 255, 0), 6)
                st.session_state.img_with_contour = cv2.cvtColor(img_draw, cv2.COLOR_BGR2RGB)
                
                st.success('✅ License plate detected! (exactly like your class code)')
                
                c1, c2 = st.columns(2)
                with c1:
                    st.image(st.session_state.img_with_contour, caption='Plate location on vehicle')
                with c2:
                    st.image(plate_cropped, caption='Cropped Plate (clio style)', channels='GRAY')
            else:
                st.error('⚠️ No plate found – try tuning parameters (common in real radars)')

# ====================== TAB 4 – BINARIZATION ======================
with tab_binarize:
    st.subheader('Binarization Algorithms (Binary + Otsu + Adaptive)')
    
    if st.session_state.plate_cropped is None:
        st.warning('Detect plate first in previous tab')
    else:
        plate = st.session_state.plate_cropped
        st.image(plate, 'Cropped Plate', channels='GRAY', use_column_width=True)
        
        bin_method = st.selectbox('Choose Binary Algorithm', 
            ['Interactive Global Threshold (your class widget)', 
             'Otsu (automatic)', 
             'Adaptive Mean', 
             'Adaptive Gaussian'])
        
        if bin_method == 'Interactive Global Threshold (your class widget)':
            thresh = st.slider('Threshold', 0, 255, 128, key='thresh_slider')
            _, binary = cv2.threshold(plate, thresh, 255, cv2.THRESH_BINARY)
            st.image(binary, f'Binary – Threshold = {thresh}', channels='GRAY', use_column_width=True)
            
        elif bin_method == 'Otsu (automatic)':
            _, binary = cv2.threshold(plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.image(binary, 'Otsu Automatic Binarization', channels='GRAY', use_column_width=True)
            
        elif bin_method == 'Adaptive Mean':
            block = st.slider('Block Size', 3, 51, 11, step=2)
            c_val = st.slider('C constant', -10, 10, 2)
            binary = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                           cv2.THRESH_BINARY, block, c_val)
            st.image(binary, f'Adaptive Mean (block={block}, C={c_val})', channels='GRAY', use_column_width=True)
            
        else:  # Adaptive Gaussian
            block = st.slider('Block Size', 3, 51, 11, step=2)
            c_val = st.slider('C constant', -10, 10, 2)
            binary = cv2.adaptiveThreshold(plate, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                           cv2.THRESH_BINARY, block, c_val)
            st.image(binary, f'Adaptive Gaussian (block={block}, C={c_val})', channels='GRAY', use_column_width=True)
        
        # Morphological cleaning (supplementary)
        if st.checkbox('Apply morphological cleaning (Opening + Closing – radar quality boost)'):
            kernel = np.ones((3,3), np.uint8)
            clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
            st.image(clean, 'Cleaned Binary (ready for OCR)', channels='GRAY', use_column_width=True)

# ====================== TAB 5 – K-MEANS ======================
with tab_kmeans:
    st.subheader('K-Means Clustering (Supplementary Segmentation)')
    
    target_img = gray
    if st.session_state.plate_cropped is not None and st.checkbox('Use cropped plate instead of full image'):
        target_img = st.session_state.plate_cropped
    
    k = st.slider('Number of clusters (K)', 2, 8, 3)
    
    if st.button('Run K-Means', type='primary'):
        Z = target_img.reshape((-1, 1)).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        
        centers = np.uint8(centers)
        segmented = centers[labels.flatten()].reshape(target_img.shape)
        
        st.image(segmented, f'K-Means Segmentation (K={k})', channels='GRAY', use_column_width=True)
        
        st.caption('Useful for color/brightness clustering before final binarization – common in advanced radar systems')

# ====================== TAB 6 – FULL PIPELINE (Radar Mode) ======================
with tab_pipeline:
    st.subheader('🚀 Full Automated Radar Pipeline')
    st.markdown('One-click simulation of a real highway radar system')
    
    if st.button('Start Full Radar Scan', type='primary', use_container_width=True):
        progress_bar = st.progress(0)
        
        # Step 1
        blur = cv2.bilateralFilter(gray, 11, 17, 17)
        progress_bar.progress(20)
        
        # Step 2
        edged = cv2.Canny(blur, 30, 200)
        progress_bar.progress(40)
        
        # Step 3 – contours
        contours, _ = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]
        plate_cnt = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            if len(approx) == 4:
                plate_cnt = approx
                break
        
        progress_bar.progress(60)
        
        if plate_cnt is not None:
            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [plate_cnt], -1, 255, -1)
            plate = cv2.bitwise_and(gray, gray, mask=mask)
            x, y, w, h = cv2.boundingRect(plate_cnt)
            plate_cropped = plate[y:y+h, x:x+w]
            st.session_state.plate_cropped = plate_cropped
            
            # Draw contour
            draw_img = img_bgr.copy()
            cv2.drawContours(draw_img, [plate_cnt], -1, (0, 255, 0), 8)
            progress_bar.progress(80)
            
            # Final binarization (Otsu + morphology)
            _, binary = cv2.threshold(plate_cropped, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)
            progress_bar.progress(100)
            
            st.success('🎉 RADAR LOCK – Plate successfully extracted!')
            
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.image(draw_img, caption='1. Detected Location', channels='BGR', use_column_width=True)
            with c2:
                st.image(plate_cropped, caption='2. Cropped Plate', channels='GRAY', use_column_width=True)
            with c3:
                st.image(binary, caption='3. Otsu Binarized', channels='GRAY', use_column_width=True)
            with c4:
                st.image(clean, caption='4. Final Cleaned Binary', channels='GRAY', use_column_width=True)
                
            st.balloons()
        else:
            st.error('No plate found with default radar parameters')

st.caption('✅ All your class code merged + powerful extras (K-means, adaptive thresholding, morphology, full pipeline). Ready for highway radar demos!')