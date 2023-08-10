#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
import os
from werkzeug.utils import secure_filename
import easyocr
import cv2
import numpy as np
import pandas as pd
import re
import string
from transformers import BertTokenizer, BertModel
import pickle
import torch
 
app = Flask(__name__)

words = ['salad','soup','chowder', 'appetizer', 'fries','strip','bowl', 'chips', 'steak', 'platter', 'pudding',\
         'chocolate','malt', 'shake','cream','creme','vanilla','brownie', 'pie', 'rings', 'wrap',\
         'juice', 'coffee', 'milk', 'tea', 'bites','drink','orange','water','burger','meat','nacho','sandwich',\
        'patty','tater','burrito','skillet','lattte','esspresso', 'cafe','sausage', 'ice cream','beer','wine']

xgb = pickle.load(open('xgb.pkl', "rb")) 
xgb2 = pickle.load(open('xgb2.pkl', "rb")) 

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a reader for English language
reader = easyocr.Reader(['en'])
 
app.secret_key = "secret key"

THUMBNAILS_FOLDER = 'static/thumbnails/'
UPLOAD_FOLDER = 'static/uploads/'
app.config['THUMBNAILS_FOLDER'] = THUMBNAILS_FOLDER
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

# Assuming these are the images available for clicking
available_images = os.listdir(UPLOAD_FOLDER)

 
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
     
@app.route('/')
def home():
    thumbnails = os.listdir(app.config['THUMBNAILS_FOLDER'])
    uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', thumbnails=thumbnails, uploaded_images=uploaded_images)
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('Image successfully uploaded and displayed below')

        thumbnails = os.listdir(app.config['THUMBNAILS_FOLDER'])
        uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])   
        
        return render_template('index.html', filename=filename, thumbnails=thumbnails, uploaded_images=uploaded_images)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)

# @app.route('/display_thumbnail/<filename>')
# def display_thumbnail(filename):
#     return redirect(url_for('static', filename='thumbnails/' + filename), code=301)
@app.route('/display//<filename>')
def display_thumbnail(filename):
    thumbnails = os.listdir(app.config['THUMBNAILS_FOLDER'])
    uploaded_images = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html', thumbnail_filename= 'thumbnails/' + filename, thumbnails=thumbnails, uploaded_images=uploaded_images)

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/display/<filename>/menu_read', methods = ['GET'])
def new_function(filename):
    print("Filename:", filename)
    def count_uppercase_letters(text):
        return sum(1 for i in text if i.isupper())
    def count_numerical_chars(text):
        return sum(1 for i in text if i.isdigit())
    def count_punctuation(text):
        return sum(1 for i in text if i in string.punctuation and i not in [',', '.', '$',':'])
    def count_consecutive_periods(text):
        return len(re.findall(r'\.{2,}', text))

    file = 'static/thumbnails/' + filename

    # Read the images using OpenCV
    image = cv2.imread(file)
    
    # Convert the image to RGB (OpenCV loads images in BGR format)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Get the dimensions of the image
    height_img, width_img, _ = image.shape
    area_img = width_img*height_img
    
    # Perform OCR on the image
    result = reader.readtext(image)

    # Prepare a list to store the data
    data = []

    # Loop through the results
    for (bbox, text, prob) in result:
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))        

        # Compute the width and height of the bounding box
        width = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)/(width_img)
        height = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)/(height_img)

        # Compute the area of the bounding box
        area = (width * height)

        # Append the text, probability, bbox coordinates, width, height, and area to the data list
        data.append([text, width, height])
    
    # Create a pandas DataFrame from the data
    df = pd.DataFrame(data, columns=["Text", "width", "height"])

    df = df[['Text', "width", 'height']]
#     df = df.astype({'width':'int'})
#     df = df.astype({'height':'int'})
#     df = df.astype({'area':'int'})
    df['uppercase'] = df['Text'].apply(count_uppercase_letters)

    df['Text'] = df['Text'].str.strip()
    # Add character count
    df['chars'] = df['Text'].apply(len)

    # Add word count
    df['words'] = df['Text'].apply(lambda x: len(x.split()))

    df['periods'] = df['Text'].apply(lambda x: x.count('.'))
    df['period_btw_numbers'] = df['Text'].apply(lambda x: bool(re.search(r'\d\.\d', x))).astype(int)
    df['number_end'] = df['Text'].apply(lambda x: bool(re.search(r'\d$', x))).astype(int)
    df['numbers'] = df['Text'].apply(count_numerical_chars)
    df['commas'] = df['Text'].apply(lambda x: x.count(','))
    df['exclamation'] = df['Text'].apply(lambda x: x.count('!'))
    df['question'] = df['Text'].apply(lambda x: x.count('?'))
    df['colons'] = df['Text'].apply(lambda x: x.count(':'))
    df['underscores'] = df['Text'].apply(lambda x: x.count('_'))
    df['dollar'] = df['Text'].apply(lambda x: x.count('$'))
    df['punctuation'] = df['Text'].apply(count_punctuation)
    df['2_periods_cnt'] = df['Text'].apply(count_consecutive_periods)

    df['Item'] = np.zeros(len(df))
    df['Item'] = df['Item'].astype('int')

    def texts_to_vectors(texts):
    
        vectors = []
        for cnt, text in enumerate(texts):
            try:            
                inputs = tokenizer([text], return_tensors="pt", truncation=True, padding=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
                mask = inputs.attention_mask
                masked_embeddings = embeddings * mask.unsqueeze(-1)
                summed = torch.sum(masked_embeddings, 1)
                summed_mask = torch.clamp(mask.sum(1), min=1e-9)
                mean_pooled = summed / summed_mask.unsqueeze(-1)
                vectors.append(mean_pooled[0].numpy())
            except Exception as e:
                print(f"Error encountered while processing text {cnt}: {e}")
                continue
            # Print progress update every 400 samples
            if cnt % 400 == 0:
                percentage = (cnt / len(texts)) * 100
                print(f"Processing text {cnt} of {len(texts)} ({percentage:.2f}% complete)")
        
        print("Finished converting texts to vectors.")

        return np.array(vectors)

    vectors = texts_to_vectors(df['Text'].tolist())    
    df_vectors = pd.DataFrame(vectors, columns=[f'vector_{i}' for i in range(vectors.shape[1])])
    df_bert_test = pd.concat([df, df_vectors], axis=1)

    df_predict = df_bert_test.drop(['Text','Item'],axis = 1)

    item = xgb.predict(df_predict)
    df_menu = df_bert_test.copy()
    df_menu['Item'] = item

    list_of_items = df_menu[df_menu.Item == 1]['Text'].str.replace('[^a-zA-Z ]', '', regex=True).str.strip().tolist()

    list_of_items_df = pd.DataFrame({'Text':list_of_items, 'Type': np.zeros(len(list_of_items),dtype = int).tolist()})

    vectors = texts_to_vectors(list_of_items_df['Text'].tolist())
    df_vectors = pd.DataFrame(vectors, columns=[f'vector_{i}' for i in range(vectors.shape[1])])
    df_bert_cat = pd.concat([list_of_items_df, df_vectors], axis=1)

    for word in words:
        df_bert_cat[word] = list_of_items_df['Text'].str.contains(word, case=False).astype(int)

    df_predict_cat = df_bert_cat.drop(['Text','Type'],axis = 1)
    Type = xgb2.predict(df_predict_cat)
    df_bert_cat['Type'] = Type
    df_final = df_bert_cat[['Text','Type']]

    categories = {
        'drinks': df_final[df_final['Type'] == 1]['Text'].tolist(),
        'appetizers': df_final[df_final['Type'] == 2]['Text'].tolist(),
        'salads': df_final[df_final['Type'] == 3]['Text'].tolist(),
        'soups': df_final[df_final['Type'] == 4]['Text'].tolist(),
        'main': df_final[df_final['Type'] == 5]['Text'].tolist(),
        'desserts': df_final[df_final['Type'] == 6]['Text'].tolist()
    }

    # for category, items in categories.items():
    #     flash(f"{category.capitalize()}: {', '.join(items)}")

    return render_template('index.html', categories=categories, filename=filename)

 
if __name__ == "__main__":
    app.run()
