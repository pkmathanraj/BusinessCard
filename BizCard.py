import pandas as pd
import streamlit as st
import easyocr
import spacy
from configparser import ConfigParser
from mysql.connector import connect, Error
from PIL import Image, ImageEnhance
import os
import re
import cv2
# from sqlalchemy import create_engine
# from matplotlib import pyplot as plt
# import nltk

st.set_page_config(
    page_title="Biz Card Extractor",
    page_icon="✅",
    layout="wide"
    )

# Extracting Database Configuration
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
        Mobile_Number VARCHAR(50),\
        EMail VARCHAR(40),\
        Website VARCHAR(40),\
        Area VARCHAR(30),\
        City VARCHAR(30),\
        State VARCHAR(40),\
        Pincode VARCHAR(10),\
        Card LONGBLOB)"
    
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
    except Error as e:
        print("Error during Table creation: ", e)

# Storing the data to the database
def save_data(texts):
    conn = dbconnection() 
    status = None
    sql = """INSERT INTO business_cards(Name, Designation, Company, Mobile_Number, EMail, Website, Area, City, State, Pincode, Card) \
        VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
    try:
        with conn.cursor() as cursor:
            cursor.execute(sql, texts)
            conn.commit()
            status = True
    except Error as e:
        print("Error during saving data to database: ", e)
        status = False
    return status

# Retrieving data from the database
def get_data():
    conn = dbconnection()
    query = "SELECT * FROM business_cards"    
    df = pd.DataFrame()
    try:
        df = pd.read_sql(query, conn)
    except Error as e:
        print("Error while extracting data from the database: ",e)
    return df

# Updating data 
def update_data(record):
    conn = dbconnection()
    query = f"""UPDATE business_cards SET Id=%s, Name=%s, Designation=%s, Company=%s,\
        Mobile_Number=%s, EMail=%s, Website=%s, Area=%s, City=%s, State=%s, Pincode=%s WHERE \
            Id={record[0]}"""
    try:
        with conn.cursor() as cursor:
            cursor.execute(query, record)
            conn.commit()
            status = True
    except Error as e:
        print("Error while updating data in the database: ",e)
        status = False
    return status

# Deleting data
def delete_data(id):
    conn = dbconnection()
    query = f"""DELETE FROM business_cards WHERE Id={id[0]}"""
    try:
        with conn.cursor() as cursor:
            cursor.execute(query)
            conn.commit()
            status = True
    except Error as e:
        print("Error while deleting data from the database: ",e)
        status = False
    return status

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

# Image Processing using pillow package
def pillow_enhancer(image_path):
    img = Image.open(image_path)
    # Adding sharpness and contrast to the image
    enhancer1 = ImageEnhance.Sharpness(img)
    enhancer2 = ImageEnhance.Contrast(img)
    img_enhance = enhancer1.enhance(20.0)
    img_enhance = enhancer2.enhance(1.5)
    return img_enhance

# Image Processing using OpenCV Package
def opencv_preprocessor(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Getting Grayscale Image
    noise = cv2.medianBlur(gray,3) # Image Noise Removal
    thresh = cv2.threshold(noise, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1] # Thresholding
    return thresh

# Marking the text detected in the image
def text_detector(result, image_path):
    image = cv2.imread(image_path)
    for detection in result:
        #get min max coordinations
        x_min, y_min = [int(cord) for cord in detection[0][0]]
        x_max, y_max = [int(cord) for cord in detection[0][2]]
        #get text
        text = detection[1]
        # declare the font
        font = cv2.FONT_HERSHEY_SIMPLEX
        # draw rectangles
        image = cv2.rectangle(image, (x_min,y_min),(x_max,y_max),(0,255,0),2)
        # put the texts
        image = cv2.putText(image, text, (x_min, y_min),font, 1, (255, 25, 200),1, cv2.LINE_AA)
    return image

# Function to extract data from business cards
@st.cache_data
def image_processor(image_path):
    # Image Processing using Pillow 
    enhanced_image = pillow_enhancer(image_path)
    enhanced_image.save("edited_image.png")

    # Image Processing using OpenCV
    processed_image = opencv_preprocessor(image_path)
    reader = easyocr.Reader(['en'])
    
    # With Pillow Enhanced Image
    results = reader.readtext("edited_image.png", paragraph=False)
    extracted_data =[result[1] for result in results]
    confidence_data = [result[2] for result in results]

    # With OpenCV Processed Image
    cvresults = reader.readtext(processed_image, paragraph=False)
    cvextracted_data =[result[1] for result in cvresults]
    cvconfidence_data = [result[2] for result in cvresults]
    
    return extracted_data, confidence_data, processed_image, cvextracted_data, cvconfidence_data, results

# Advanced Text Processor using Spacy - combined tokens
@st.cache_data
def text_processor_advance_join(txt_lst):
    with st.spinner():
        # Here joining all the texts as a single statement
        remaining_text = " ".join([txt for txt in txt_lst])
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(remaining_text)
        for i in doc.ents:
            st.write(i.text, i.label_)
        # Initialize variables to store identified entities
        person_name = []
        company_name = []
        designation = []

        # Iterate through spaCy entities and classify them
        for ent in doc.ents:
            if ent.label_ == "PERSON":
                person_name.append(ent.text)
            elif ent.label_ == "ORG":
                company_name.append(ent.text)
            elif ent.label_ == "TITLE":
                designation.append(ent.text)
        st.divider()
        st.write("Person Name: ",person_name)
        st.write("Company Name: ",company_name)
        st.write("Designation: ",designation)

# Advanced Text Processor using Spacy - for separate tokens
@st.cache_data
def text_processor_advance(txt_lst):
    with st.spinner():
        # Here - testing with text as a separate token        
        nlp = spacy.load("en_core_web_sm")
        doc = []
        for txt in txt_lst:
            doc.append(nlp(txt))        
        # Initialize variables to store identified entities
        person_name = []
        company_name = []
        designation = []

        # Iterate through spacy entities and classify them
        for d in doc:
            for ent in d.ents:
                if ent.label_ == "PERSON":
                    person_name.append(ent.text)
                elif ent.label_ == "ORG":
                    company_name.append(ent.text)
                elif ent.label_ == "TITLE":
                    designation.append(ent.text)
        st.divider()
        # st.write("Person Name: ",person_name)
        # st.write("Company Name: ",company_name)
        # st.write("Designation: ",designation)

# Function to process extracted text
@st.cache_data
def text_processor(text_lst):        
    processed_text = {"Name": None,
                      "Designation": None,
                      "Company": None,
                      "Mobile_number": None,
                      "Email": None,
                      "Website": None,
                      "Area": None,
                      "City": None,
                      "State": None,
                      "Pincode": None,
                      "Card": None
                      }
    remaining_text=[]
    email_regex = "([a-zA-Z][\w\-\.]+)@([\w\-\.]+)\.?([a-zA-Z]{2,5})$"
    web_regex = r"[wW]{3}[\s\.]?\w+[-\.\s]?\w+[\.\s]?\w+[\.\s]?\w+"
    phone_no_regex = r"(\+?\d{2,3}\s?[-]?\d{3,5}\s?[-]?\d{4,5})"
    pincode_regex = r"\d{3}\s?\d{1,3}"
    area_regex = r"(St\.)$"
    city_regex = r"(City)$"
    
    # Compile the ReGex 
    email = re.compile(email_regex)
    pincode = re.compile(pincode_regex)
    inter_ph = re.compile(phone_no_regex)
    web = re.compile(web_regex)
    area = re.compile(area_regex)
    city = re.compile(city_regex)
    phone_numbers = []

    # Update the following with NER using SpaCy
    # Temporary Usage - Not Effective for all business card types
    processed_text['Name'] = text_lst[0]
    processed_text['Designation'] = text_lst[1]
    processed_text['Company'] = text_lst[2]
    processed_text['State'] = text_lst[-1]
    processed_text['City'] = text_lst[-2]
    processed_text['Area'] = text_lst[-3]

    with st.spinner(): 
        for txt in text_lst:            
            mail = re.search(email, txt.replace(' ','')) 
            pin = re.search(pincode, txt) 
            phno = re.search(inter_ph, txt) 
            website = re.search(web, txt.lower())
            area_txt = re.search(area, txt)
            city_txt = re.search(city, txt)
            
            if mail:                
                email_id = mail.group()
                if not re.search(re.compile(r"\.\w{2,3}$"), email_id): # To check for missing . before domain name at the end
                    end_txt = re.search(r"\w{2,3}$", email_id).group()
                    email_id = email_id.replace(end_txt,f'.{end_txt}') # To add missing . before domain name
                processed_text['Email'] =  email_id

            elif phno:
                phone_numbers.append(phno.group())            
            
            elif website:
                txt = txt.lower().replace(' ','.') # Replace all the spaces with . which is usually after www or before domain name
                if re.search(re.compile(r"[wW]{3}\w+"), txt): # To check for missing . after www
                    txt = txt.replace('www','www.') # To add missing . after www                
                if not re.search(re.compile(r"\.\w{2,3}$"), txt): # To check for missing . before domain name at the end
                    end_txt = re.search(r"\w{2,3}$", txt).group()
                    txt = txt.replace(end_txt,f'.{end_txt}') # To add missing . before domain name
                processed_text['Website'] = txt
            
            elif pin:
                # txt = re.sub(r'[\s\._]', '', txt) # Replacing / Substituting any single space or . or _ to nothing (Removing single space or . or _)
                # if len(txt) == 6:
                processed_text['Pincode'] = pin.group()

            elif 'St' or 'ST' or 'Str' or 'STR' or 'Street' or 'street' in txt:
                processed_text['Area'] = txt
            
            elif 'City' or 'city' in txt:
                processed_text['City'] = txt
            
            elif txt:
                remaining_text.append(txt)
        if phone_numbers:
                processed_text['Mobile_number'] = phone_numbers
    # st.write(remaining_text)
    # st.write(processed_text)
    text_processor_advance(remaining_text)
    return processed_text

# Converting Image file to Binary Format to store in MySQL
@st.cache_resource
def image_to_binary(file):
    with open(file, 'rb') as file:
        binary_data = file.read()
    return binary_data

# Front End Design
def front_end():
    st.header("Business Card Extractor", divider="rainbow")
    task = st.sidebar.radio(label = "Select the Task:", options = ["Extract", "View", "Modify"])
    
    if task == "Extract":
        # File uploader to upload the image
        uploaded_image = st.file_uploader("Select a Business Card Image to Upload:", type=['jpg', 'jpeg', 'png'])
        extracted_text = None

        if uploaded_image is not None:
            col1, col2, col3 = st.columns([2,1,0.5])
            # Displaying the uploaded image
            col1.image(uploaded_image, caption="Uploaded Image", width=500)
            
            # Getting the image upload path 
            image_filename = os.path.join("images",uploaded_image.name)
            with open(image_filename, "wb") as f:
                f.write(uploaded_image.read())
            
            with st.spinner("Extracting Data from the Uploaded Image..."):
                # Image Preprocessing before text extraction
                extracted_text, confidence_data, processed_img, cvextracted, cvconfidence, result = image_processor(image_filename)
                
                # To display the processed image using pillow package
                # col1.image("edited_image.png", caption="Pillow Enhanced Image", width=500)

                # To display the processed image using opencv package
                # col1.image(processed_img, caption="OpenCV Processed Image", width=500)
                
                # Calling the function to detect the text
                txt_detected_img = text_detector(result, image_filename)

                # Displaying the image with detected text marked in a box
                col2.image(txt_detected_img, caption="Detected Text - Image", width=500)   

                # Displaying the extracted texts
                col1.subheader("Extracted Info :")
                col1.write("With Pillow Enhanced Image:")
                col1.write(extracted_text)

                # Displaying the Detected texts and Confidence Data using Pillow and OpenCV
                # col2.write("With OpenCV Processed Image:")
                # col2.write(cvextracted)
                # col3.subheader("Confidence Level :")
                # col3.write("With Pillow Enhanced Image:")
                # col3.write(confidence_data)
                # col3.write("With OpenCV Processed Image:")
                # col3.write(cvconfidence)

            text_to_store = text_processor(extracted_text) # Processing the detected text to identify the type of entity - dict obj 
            binary_image = image_to_binary(image_filename) # Converting the image to binary form to store in db

            # Extracting the keys and values from the result dictionary          
            keys = list(text_to_store.keys())
            values = list(text_to_store.values())

            # Displaying all the extracted text in a text input and omitting the last binary image value
            data_tuple=()            
            for name, txt in zip(keys[:-1], values[:-1]):            
                edited_value = col2.text_input(label=name, value=txt)
                data_tuple += (edited_value,)
            submit = col2.button("Save to Database", type="primary")
            col2.caption("Edit the values if needed before saving.")
            data_tuple += (binary_image,)
            if submit:
                status = save_data(data_tuple)
                if status:
                    st.toast("Data Successfully stored in the database")
                else:
                    st.toast("Data Failed to store in the database")

    elif task=="View":
        df = get_data()
        st.subheader("Business Card Details: ")
        df.index = df.index + 1        
        df.index.name = "#"
        st.write(df)

    elif task == "Modify":
        task_choice = st.radio("Select the operation to perform", ["Edit/Update", "Delete"], horizontal=True)        
        if task_choice == "Edit/Update":           
            st.subheader("Edit/Update the existing Data")
            df = get_data()        
            options = df['Name']
            choice = st.selectbox("List of available Business Cards: ", options)      
            st.divider()  
            col1, col2 = st.columns(2)
            col1.subheader("Data: ", divider="violet")
            modified_values=()
            result = df[df['Name'] == choice].drop(columns='Card').iloc[0]              
            for i, (column, value) in enumerate(result.items()):
                key = f"txt_input_{i}"
                disabled = i==0
                modified_data = col1.text_input(label=column, value=value, key=key, disabled=disabled)
                modified_values += (modified_data,)
            update = st.button("Update", type="primary")
            if update:
                status = update_data(modified_values)
                if status:
                    st.toast("Data Successfully updated in the database")
                else:
                    st.toast("Data Failed to update in the database")
            img_df = df[df['Name'] == choice]
            img = tuple(img_df['Card'])
            col2.subheader("Business Card Image: ", divider="violet")
            col2.image(img[0])

        elif task_choice == "Delete":            
            st.subheader("Delete the existing Data")
            df = get_data()        
            options = df['Name']
            choice = st.selectbox("List of available Business Cards: ", options)
            st.divider()            
            col1, col2 = st.columns(2)
            col1.subheader("Selected Data: ", divider="violet")
            result_df = df[df['Name'] == choice]
            col1.dataframe(result_df)
            selected = tuple(result_df['Id'])
            delete = st.button("Delete", type="primary")
            if delete:
                status = delete_data(selected)
                if status:
                    st.toast("Data Successfully deleted from the database")
                else:
                    st.toast("Data Failed to delete from the database")
                
            img_df = df[df['Name'] == choice]
            img = tuple(img_df['Card'])
            col2.subheader("Business Card Image: ", divider="violet")
            col2.image(img[0])

# Function to display the image from the binary data 
def display_image(binary_data):
    st.image(binary_data, use_column_width=True)
    
# Main Function 
def main():    
    # Create the folder if it doesn't exist
    if not os.path.exists("images"):
        os.makedirs("images")

    create_mysqlschema()
    front_end()
    
if __name__ == "__main__":
    main() 
