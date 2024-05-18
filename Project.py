import pandas as pd
import numpy as np
import cv2, re
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
from sklearn import metrics, tree
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

## Streamlit Page config
st.set_page_config(page_title= 'E-Commerce', layout= 'wide', page_icon='chart_with_upwards_trend', initial_sidebar_state='expanded')

option = option_menu("", ["EDA", "Model", "Image processing", "NLP", "Recommendation"], icons=['search', 'brush', 'calculator', 'gear', 'graph-up'], default_index=0, orientation="horizontal")

# Data Access
data = pd.read_csv("C:/Users/user/Desktop/Guvi/Dataset/New folder/E-commerce.csv")
df = pd.read_csv("C:/Users/user/Desktop/Guvi/Dataset/New folder/E-commerce.csv")
new_num = ['count_session','count_hit','geoNetwork_region','geoNetwork_latitude','historic_session_page','historic_session','geoNetwork_longitude','last_visitId','sessionQualityDim','single_page_rate','avg_session_time_page','avg_session_time','time_earliest_visit','latest_visit_number','earliest_visit_number','earliest_visit_id','visitId_threshold','latest_visit_id','avg_visit_time','time_latest_visit','bounce_rate','visits_per_day','transactionRevenue','time_on_site','num_interactions','transactionRevenue','time_on_site']
cat = data.select_dtypes(include='object').columns
data = data.drop_duplicates()
data = data.reset_index(drop=True)
lb = LabelEncoder()

for i in cat:
    data[i] = lb.fit_transform(data[i])

# EDA
if option == 'EDA':
    st.markdown('## :red[Before Data Cleaning]')
    st.write(df.head(), '\n #### :blue[Shape of Dataset]', df.shape)
    st.markdown('## :red[After Data Cleaning]')
    st.write(data.head(), '\n #### :blue[Shape of Dataset]', data.shape)

    sel = st.radio(label= "## :red[Select the input]", options = [':green[Head]',':green[Tail]',':green[Duplicates]',':green[Shape]',':green[Describe]',':green[Columns]',':green[Null Value]',':green[Correlation]'],label_visibility='visible', horizontal=True, index=7)
    if sel == ':green[Head]':
        st.write(':blue[Head : ]')
        st.write(data.head())
    if sel == ':green[Tail]':
        st.write(':blue[Tail : ]')
        st.write(data.tail())
    if sel == ':green[Duplicates]':
        st.write(':blue[Total Duplicates : ]')
        st.write(data.duplicated().value_counts())
        st.write(df.shape)
    if sel == ':green[Shape]':
        st.write(':blue[Shape : ]')
        st.write(data.shape)
    if sel == ':green[Describe]':
        st.write(':blue[Describe : ]')
        st.write(data.describe())
    if sel == ':green[Columns]':
        st.write(':blue[Columns : ]')
        st.write(data.columns)
        st.write(':blue[Total No. of Columns :] ', data.shape[1])
    if sel == ':green[Null Value]':
        st.write(':blue[Null Values : ]')
        st.write(data.isnull().sum().reset_index())
        st.write(':blue[Total No. of Null values :] ', data.isnull().sum().sum())
    if sel == ':green[Correlation]':
        st.write(':blue[Correlation : ]')
        st.write(data.corr())
    plots = st.selectbox("## :red[Select Your Plot type]", options=['Hist Plot','Relationship Plot','Heat Map','Distribution Plot'])
    x = st.selectbox("# :violet[Select the x-axis]", new_num)
    y = st.selectbox("# :violet[Select the y-axis]", new_num)
    fig, ax = plt.subplots()
    if plots == 'Hist Plot':
        plt.title('Hist Plot')
        sns.histplot(data= data, x=data[x], color = 'r',bins = 20,element='bars', kde=True, linewidth = 1.2, edgecolor ='black')
        plt.tight_layout()
        plt.grid(True, linestyle = "--")
        plt.xlabel(x)
        st.pyplot(fig)
    if plots == 'Relationship Plot':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.title('Relationship Plot')
        sns.relplot(data, x = data[x], y = data['has_converted'], kind= 'scatter')
        plt.xlabel(x)
        plt.ylabel('Counts')
        st.pyplot()
    if plots == 'Heat Map':
        st.set_option('deprecation.showPyplotGlobalUse', False)
        plt.figure(figsize=(25,15))
        sns.heatmap(data.corr(numeric_only=True), vmin = -1, vmax =1,annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Matrix')
        plt.tight_layout()
        st.pyplot()
    if plots == 'Distribution Plot':
        sns.displot(data, label = x)
        plt.title("Distribution Plot")
        plt.legend()
        st.pyplot()

# Model Building
if option == 'Model':
    st.write(data.head())
    st.write(':blue[Shape of the Data : ]', data.shape)
    x = data.drop('has_converted', axis=1)
    y = data[['has_converted']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=90)
    def mod(pred):
        st.write(":orange[Accuracy Score: ]",metrics.accuracy_score(y_test, pred))
        st.write(":red[Recall Score: ]", metrics.recall_score(y_test, pred))
        st.write(":green[Precision Score:] ",metrics.precision_score(y_test, pred))
        st.write(":violet[F1 Score: ]", metrics.f1_score(y_test, pred))
        st.write(":rainbow[Confusion Matrix:] ", metrics.confusion_matrix(y_test, pred))
    model = st.selectbox(':blue[Select the Model : ]', options = ['Logistic Regression', 'Decision Tree', 'KNN'])

    if model == 'Logistic Regression':
        lr = LogisticRegression()
        lr_model = lr.fit(x_train, y_train)
        ypred = lr_model.predict(x_test)
        mod(ypred)
    if model == 'Decision Tree':
        dt = DecisionTreeClassifier()
        dt_model = dt.fit(x_train, y_train)
        ypred = dt_model.predict(x_test)
        mod(ypred)
    if model == 'KNN':
        kn = KNeighborsClassifier()
        kn_model = kn.fit(x_train, y_train)
        ypred = kn_model.predict(x_test)
        mod(ypred)

# Image Processing
if option == 'Image processing':
    img = st.file_uploader('#### :red[Upload Image]', type=['png', 'jpg', 'jpeg'], label_visibility='visible', accept_multiple_files=False)
    if img is not None:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(':blue[Normal Image]')
            st.image(img, caption='Normal Image')
            st.write(':blue[Grayscale Image]')
            file = np.array(bytearray(img.read()), dtype=np.uint8)
            img = cv2.imdecode(file, cv2.IMREAD_COLOR)
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            st.image( gray_img,caption = 'Grayscale Image')
        with col2:
            st.write(':blue[Blur Image]')
            blur = cv2.GaussianBlur(gray_img, (11,11),0)
            st.image(blur, caption='Blur Image')
            st.write(":blue[Edge Detect Image]")
            edges = cv2.Canny(blur, 50,150)
            st.image(edges, caption = "Edge Detect Image")
        with col3:
            st.write(':blue[De-Blur Image]')
            psf = np.ones((5, 5)) / 25
            deb_img = cv2.filter2D(blur, -1, psf)
            st.image(deb_img,caption='De-Blur Image')
            st.write(":blue[Sharp Image]")
            kernel = np.array([[-1, -1, -1],
                            [-1, 9, -1],
                            [-1, -1, -1]])
            sharpened = cv2.filter2D(img, -1, kernel)
            st.image(sharpened, caption='Sharp Image')

# NLP
if option == 'NLP':
    import nltk
    from nltk import ne_chunk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.stem.porter import PorterStemmer
    from nltk.tokenize import word_tokenize, sent_tokenize
    from nltk.corpus import stopwords
    from nltk.probability import FreqDist
    from wordcloud import WordCloud 
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('words')
    nltk.download('vader_lexicon') 
    nltk.download('maxent_ne_chunker')
    nltk.download('punkt')
    txt = st.text_input("#### :red[Type a Text]")
    st.write(txt)

    # Cleaning & Analysing Text
    cln = st.selectbox('#### :red[Select Category]', ['Cleaning', 'Analyse'])

    #Cleaning
    if cln == 'Cleaning':
        opt = st.radio('#### :red[Select the Option]', options = [':blue[Stopwords]', ':blue[Stemming]', ':blue[Word_Tokens]', ':blue[Sent_Tokens]'], horizontal=True)
        txt = re.sub(r'\W+', ' ', txt).lower()
        txt = txt.replace("[^a-zA-Z]", " ")        
        if opt == ':blue[Stopwords]':
            words = word_tokenize(txt)
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            st.write(filtered_words)
        if opt == ':blue[Stemming]':
            ps = PorterStemmer()
            stem = [ps.stem(word) for word in word_tokenize(txt)]
            st.write(stem)
        if opt == ':blue[Word_Tokens]':
            word_tokens = word_tokenize(txt) 
            st.write(word_tokens)
        if opt == ':blue[Sent_Tokens]':
            sent_tokens = sent_tokenize(txt)
            st.write(sent_tokens)
    
    # Analysing
    if cln == 'Analyse':
        opt = st.radio('#### :red[Select the Option]', options = [':blue[POS]', ':blue[NER]', ':blue[Keyword_Extract]', ':blue[WordCloud]', ':blue[Sentiment_Analysis]'], horizontal=True)
        if opt == ':blue[POS]':
            pos = word_tokenize(txt)
            token = []
            for tok in pos:
                token.append(nltk.pos_tag([tok]))
            st.write(token)
        if opt == ':blue[NER]':
            ner = word_tokenize(txt)
            named_entities = []
            ne_chunked = ne_chunk(nltk.pos_tag(ner))
            for chunk in ne_chunked:
                if hasattr(chunk, 'label'):
                    entity = ' '.join(c[0] for c in chunk)
                    entity_type = chunk.label()
                    named_entities.append((entity, entity_type))
            st.write(':orange[Named Entities : ]')
            for entity, entity_type in named_entities:
                st.write(f"{entity} - {entity_type}")
        if opt == ':blue[Keyword_Extract]':
            sentences = sent_tokenize(txt)
            words = word_tokenize(txt)
            stop_words = set(stopwords.words('english'))
            filtered_words = [word for word in words if word.lower() not in stop_words]
            filtered_words = [''.join(char for char in word if char.isalpha()) for word in filtered_words]
            word_frequencies = FreqDist(filtered_words)
            sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [word for word, freq in sorted_word_frequencies[:10]]
            st.write(":orange[Top Keywords : ]", len(top_keywords))
            for keyword in top_keywords:
                st.write(keyword)
        if opt == ':blue[WordCloud]':
            wordcloud = WordCloud(width=800, height = 400, background_color='white').generate(txt)
            plt.figure(figsize=(10, 5))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot()
        if opt == ':blue[Sentiment_Analysis]':
            sia = SentimentIntensityAnalyzer()
            sia_score = sia.polarity_scores(txt)
            st.write(':orange[Sentiment Score : ]', sia_score)
            if sia_score['compound'] >= 0.05:
                st.write(':green[Overall Sentiment : Positive]')
            elif sia_score['compound'] <= -0.05:
                st.write(':green[Overall Sentiment : Negative]')
            else:
                st.write(':green[Overall Sentiment : Neutral]')

if option == 'Recommendation':
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    user_input = st.text_input("##### :red[Enter the name of the product: ]")
    data = {
        'product_id': [1, 2, 3, 4, 5],
        'product_name': ['Laptop', 'Smartphone', 'Headphones', 'Tablet', 'Camera'],
        'description': ['Powerful laptop with high-performance specs', 
                        'Latest smartphone with advanced features', 
                        'Premium headphones for immersive audio experience', 
                        'Compact tablet for on-the-go productivity', 
                        'High-resolution camera for professional photography']}
    if st.button(':orange[Get Recommendation]'):
        df = pd.DataFrame(data)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(df['description'])
        cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
        try:
            def get_recommendations(product_name, cosine_sim=cosine_sim):
                idx = df[df['product_name'] == product_name].index[0]
                sim_scores = list(enumerate(cosine_sim[idx]))
                sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
                sim_scores = sim_scores[1:6]
                product_indices = [i[0] for i in sim_scores]
                return df['product_name'].iloc[product_indices]
            recommendations = get_recommendations(user_input)
            st.write(":blue[Recommended Products: ]", recommendations)
        except :
            st.write(":blue[Please enter the word properly]")


    


        




        


        

    
        