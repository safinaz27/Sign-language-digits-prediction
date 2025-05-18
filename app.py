import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# تحميل النموذج المدرب
model = tf.keras.models.load_model('model.h5')  # غيّري الاسم حسب اسم الملف بتاعك

# عناوين الليبلز (مثلاً من 1 إلى 10)
labels_map = {
    0: '9',
    1: '7',
    2: '6',
    3: '8',
    4: '1',
    5: '0',
    6: '4',
    7: '3',
    8: '2',
    9: '5'
}

st.title("Sign language digits prediction")

uploaded_file = st.file_uploader("upload image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # عرض الصورة
    
    image = Image.open(uploaded_file) 
    st.image(image, caption='uploded image', width=150)
    image = image.convert('L')# تحويل لصورة رمادية
    image = image.resize((64, 64))  # تغيير الحجم
    # st.image(image, caption='الصورة المدخلة', width=150)

    # تحضير الصورة للنموذج
    img_array = np.array(image) # تطبيع القيم
    img_array = img_array.reshape(1, 64, 64, 1)  # الشكل المناسب للموديل

    # عمل التنبؤ
    # predicted_classes = np.argmax(model.predict(X_val), axis=1)
    prediction = model.predict(img_array)
    pred_class = np.argmax(prediction)

    st.write("### Prediction :")
    st.write(f"number **{pred_class}**")
