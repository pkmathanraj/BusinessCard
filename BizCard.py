import pandas as pd
import streamlit as st
import easyocr
import nltk
import spacy
from configparser import ConfigParser
from mysql.connector import connect, Error
from sqlalchemy import create_engine
from PIL import Image, ImageEnhance
import os
import re
import cv2

st.set_page_config(
    page_title="Biz Card Extractor",
    page_icon="✅",
    layout="wide"
    )

@st.cache_data
def config(filename='database.ini', section='mysql'):
    parser = ConfigParser()
    parser.read(filename)
  
    # get section, default to mysql
    conn_param = {}
    if parser.has_section(section):
        params = parser.items(section)
        for p in params:
            conn_param[p[0]] = p[1]
    else:
        raise Exception('Section {0} not found in the {1} database config file'.format(section, filename))
  
    return conn_param

# Establishing MySQL Connection
@st.cache_resource
def dbconnection():
    params = config()
    try:
        conn = connect(**params)
        with conn.cursor() as cursor:
            cursor.execute("CREATE DATABASE IF NOT EXISTS BizCard")
        params['database']='BizCard'
        return connect(**params)       
    
    except Error as e:
        print("Error during establishing MySQL connection: ",e)


# Creating tables in MySQL
def create_mysqlschema():
    conn = dbconnection()
    query = "CREATE TABLE IF NOT EXISTS business_cards(\
        Id INTEGER PRIMARY KEY AUTO_INCREMENT,\
        Name VARCHAR(40),\
        Designation VARCHAR(40),\
        Company VARCHAR(40),\
        Mobile_Number VARCHAR(10),\
        EMail VARCHAR(40),\
        Website VARCHAR(40),\
        Area VARCHAR(30),\
        City VARCHAR(30),\
        State VARCHAR(30),\
        Pincode VARCHAR(6),\
        Card LONGBLOB)"
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
                
    except Error as e:
        print("Error during Table creation: ",e)

# Check for dark background
def is_dark_background(image_path, threshold=100):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate the average color intensity of the image
    average_color = cv2.mean(gray_image)[0]

    # Compare the average color with the threshold to determine if it's a dark background
    if average_color < threshold:
        return True
    else:
        return False

def pillow_enhancer(image_path):
    img = Image.open(image_path)
    # Adding sharpness and contrast to the image
    enhancer1 = ImageEnhance.Sharpness(img)
    enhancer2 = ImageEnhance.Contrast(img)
    img_enhance = enhancer1.enhance(20.0)
    img_enhance = enhancer2.enhance(1.5)
    img_enhance.save("edited_image.png")

def opencv_preprocessor(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    noise = cv2.medianBlur(gray,3)
    thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh

# Function to extract data from business cards
# @st.cache_data
def image_processor(image_path):
    is_bgdark = is_dark_background(image_path)
    if is_bgdark:
        print("Dark Background")
    else:
        print("Light Background")

    pillow_enhancer(image_path)
    processed_image = opencv_preprocessor(image_path)
    reader = easyocr.Reader(['en'])
    #image = cv2.imread(image_path)
    # With Pillow Enhanced Image
    results = reader.readtext("edited_image.png", paragraph=False)
    extracted_data =[result[1] for result in results]
    confidence_data = [result[2] for result in results]

    # With OpenCV Processed Image
    cvresults = reader.readtext(processed_image, paragraph=False)
    cvextracted_data =[result[1] for result in cvresults]
    cvconfidence_data = [result[2] for result in cvresults]
    return extracted_data, confidence_data, processed_image, cvextracted_data, cvconfidence_data

@st.cache_data
def text_processor_advance(txt_lst):
    with st.spinner():
        remaining_text = " ".join([txt for txt in txt_lst])
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(remaining_text)
        for i in doc:
            st.write(i.text, i.pos_)

# Function to process extracted text
# @st.cache_data
def text_processor(text_lst):    
    processed_text={}
    remaining_text=[]
    email_regex = "^([a-zA-Z0-9_\-\.]+)@([a-zA-Z0-9_\-\.]+)\.([a-zA-Z]{2,5})$"
    pincode_regex = "^[1-9]{1}[0-9]{2}\\s{0,1}[0-9]{3}$"
    international_ph_no_regex = "^[+]{1}(?:[0-9\\-\\(\\)\\/\\.]\\s?){6, 15}[0-9]{1}$"
    web_regex = "^((?!-)[A-Za-z0-9-]" + "{1,63}(?<!-)\\.)" +"+[A-Za-z]{2,6}"
    # Compile the ReGex 
    email = re.compile(email_regex)
    pincode = re.compile(pincode_regex)
    inter_ph = re.compile(international_ph_no_regex)
    web = re.compile(web_regex)
    with st.spinner():
        for txt in text_lst:
            mail = re.match(email, txt)           
            #pin = re.match(pincode, txt) 
            #phno = re.search(inter_ph, txt)

            if mail != None:                
                processed_text['email'] =  txt
                #st.write(txt)
            elif re.search(pincode, txt):
                processed_text['pincode-reg'] = txt
            elif re.search(inter_ph, txt):
                processed_text['phone_no-reg'] = txt
            elif re.search(web, txt):
                processed_text['website-reg'] = txt


            if 'www' or 'WWW' in txt:
                processed_text['website'] = txt
            elif (txt.replace(" ","")).isdigit() and len(txt.replace(" ",""))>=10:
                processed_text['mobile'] = txt.replace(" ","")
            elif (txt.replace(" ","")).isdigit() and len(txt.replace(" ",""))==6 or len(txt.replace(" ",""))==5:
                processed_text['pincode'] = txt.replace(" ","")
            elif 'St' or 'ST' or 'Str' or 'STR' or 'Street' or 'street' in txt:
                processed_text['area'] = txt
            elif txt:
                remaining_text.append(txt)
    st.write(remaining_text)
    st.write(processed_text)
    text_processor_advance(remaining_text)

        


# Converting Image file to Binary Format to store in MySQL
@st.cache_resource
def image_to_binary(file):
    with open(file, 'rb') as file:
        binary_data = file.read()
    return binary_data

def front_end():
    st.header("Business Card Extractor", divider="rainbow")
    task = st.sidebar.radio(label = "Select the Task:", options = ["Extract", "View", "Modify"])
    
    if task=="Extract":
        uploaded_image = st.file_uploader("Select a Business Card Image to Upload:", type=['jpg', 'jpeg', 'png'])
        extracted_text = None
        if uploaded_image is not None:
            col1, col2, col3 = st.columns([2,1,1])
            col1.image(uploaded_image, caption="Uploaded Image", width=500)
            image_filename = os.path.join("images",uploaded_image.name)
            with open(image_filename, "wb") as f:
                f.write(uploaded_image.read())

            with st.spinner("Extracting Data from the Uploaded Image..."):
                extracted_text, confidence_data, processed_img, cvextracted, cvconfidence = image_processor(image_filename)
                col1.image("edited_image.png", caption="Pillow Enhanced Image", width=500)
                col1.image(processed_img, caption="OpenCV Processed Image", width=500)
                col2.subheader("Extracted Info :")
                col2.write("With Pillow Enhanced Image:")
                col2.write(extracted_text)
                col2.write("With OpenCV Processed Image:")
                col2.write(cvextracted)
                col3.subheader("Confidence Level :")
                col3.write("With Pillow Enhanced Image:")
                col3.write(confidence_data)
                col3.write("With OpenCV Processed Image:")
                col3.write(cvconfidence)
            text_processor(extracted_text)
            
    
# Main Function 
def main():
    bgpath="bg1.jpg"
    bgstyle=f"""body{{background-image: url('{bgpath}'),
                     background-size: cover,
                     background-repeat: no-repeat,
                     background-attachment: fixed    }}"""
    print(bgstyle)
    st.markdown(f"<style> {bgstyle}</style>", unsafe_allow_html=True)
    # Create the folder if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    create_mysqlschema()
    front_end()
    
if __name__ == "__main__":
    main() 