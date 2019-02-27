from keras.models import Model, Sequential
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Concatenate, InputLayer, Reshape, BatchNormalization, LSTM, Lambda
import json
import sys
from PIL import Image
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def get_cross_dim (embeddingsType):
    dim = 300
    if (embeddingsType == "scratch"):
        dim = int(dim / 1)
    if (embeddingsType == "fastText"):
        dim = int(dim / 2)
    if (embeddingsType == "holE" or embeddingsType == "wikipedia" or embeddingsType == "semantic-scholar"):
        dim = int(dim / 3)
    return dim

def get_cat_dim (weights):
    dim = 300
    if (weights == "cross"):
        dim = int(dim / 1)
    if (weights == "cross-vecsi"):
        dim = int(dim / 3)
    return dim


def get_n_images (corpus):
    n_images = 0
    if corpus == "scigraph":
        n_images = n_images + 82396*2
    if corpus == "semantic-scholar":
        n_images = n_images + 476746*2
    return n_images

def get_cross_model (embeddingsType, word_index_tokens, word_index_syncons, embedding_matrix_tokens, embedding_matrix_syncons, max_sequence_length, dim):
    if (embeddingsType == "scratch"):
        modelCaptions = Sequential()
        modelCaptions.add(Embedding(len(word_index_tokens)+1, dim, embeddings_initializer="uniform", input_length=max_sequence_length,trainable=True))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(5))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(5))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(35))
        modelCaptions.add(Reshape((1,1,512)))
      
        modelImages = Sequential()
        modelImages.add(InputLayer(input_shape=(224,224,3)))
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D((28,28),2))        
      
        mergedOut = Concatenate()([modelCaptions.output,modelImages.output])
        mergedOut = Flatten()(mergedOut)    
        mergedOut = Dense(128, activation='relu')(mergedOut)
        mergedOut = Dense(2, activation='softmax')(mergedOut)
        
        model = Model([modelCaptions.input, modelImages.input], mergedOut)

    if (embeddingsType == "fastText"):
        modelCaptionsScratch = Sequential()
        modelCaptionsScratch.add(Embedding(len(word_index_tokens)+1, dim, embeddings_initializer="uniform", input_length=max_sequence_length,trainable=True))
        modelCaptionsVecsiTokens = Sequential()
        modelCaptionsVecsiTokens.add(Embedding(len(word_index_tokens) + 1,dim, weights = [embedding_matrix_tokens], input_length = max_sequence_length, trainable = False))
        modelMergeEmbeddings = Concatenate()([modelCaptionsScratch.output,modelCaptionsVecsiTokens.output])
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(35)(modelMergeEmbeddings)
        modelMergeEmbeddings = Reshape((1,1,512))(modelMergeEmbeddings)
        modelCaptions = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input], modelMergeEmbeddings)
      
        modelImages = Sequential()
        modelImages.add(InputLayer(input_shape=(224,224,3)))
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D((28,28),2))        
      
        mergedOut = Concatenate()([modelCaptions.output,modelImages.output])
        mergedOut = Flatten()(mergedOut)    
        mergedOut = Dense(128, activation='relu')(mergedOut)
        mergedOut = Dense(2, activation='softmax')(mergedOut)
        
        model = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input,modelImages.input], mergedOut)

    if (embeddingsType == "holE" or embeddingsType == "wikipedia" or embeddingsType == "semantic-scholar"):
        modelCaptionsScratch = Sequential()
        modelCaptionsScratch.add(Embedding(len(word_index_tokens)+1, dim, embeddings_initializer="uniform", input_length=max_sequence_length,trainable=True))
        modelCaptionsVecsiTokens = Sequential()
        modelCaptionsVecsiTokens.add(Embedding(len(word_index_tokens) + 1,dim, weights = [embedding_matrix_tokens], input_length = max_sequence_length, trainable = False))
        modelCaptionsVecsiSyncons = Sequential()
        modelCaptionsVecsiSyncons.add(Embedding(len(word_index_syncons) + 1,dim, weights = [embedding_matrix_syncons], input_length = max_sequence_length, trainable = False))
        modelMergeEmbeddings = Concatenate()([modelCaptionsScratch.output,modelCaptionsVecsiTokens.output,modelCaptionsVecsiSyncons.output])
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(35)(modelMergeEmbeddings)
        modelMergeEmbeddings = Reshape((1,1,512))(modelMergeEmbeddings)
        modelCaptions = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input,modelCaptionsVecsiSyncons.input], modelMergeEmbeddings)
      
        modelImages = Sequential()
        modelImages.add(InputLayer(input_shape=(224,224,3)))
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D(2))
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
        modelImages.add(BatchNormalization())
        modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
        modelImages.add(BatchNormalization())
        modelImages.add(MaxPooling2D((28,28),2))        
      
        mergedOut = Concatenate()([modelCaptions.output,modelImages.output])
        mergedOut = Flatten()(mergedOut)    
        mergedOut = Dense(128, activation='relu')(mergedOut)
        mergedOut = Dense(2, activation='softmax')(mergedOut)
        
        model = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input,modelCaptionsVecsiSyncons.input,modelImages.input], mergedOut)
    return model
def get_cat_captions_model (weights, word_index_tokens, word_index_syncons, max_sequence_length, dim):
    if (weights == "cross"):
        modelCaptions = Sequential()
        modelCaptions.add(Embedding(len(word_index_tokens)+1, dim, embeddings_initializer="uniform", input_length=max_sequence_length,trainable=True))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(5))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(5))
        modelCaptions.add(Conv1D(512, 5, activation="relu"))
        modelCaptions.add(MaxPooling1D(35))
        modelCaptions.add(Reshape((1,1,512)))
        modelCaptions.add(Flatten())
        modelCaptions.add(Dense(128, activation='relu'))
        modelCaptions.add(Dense(5, activation='softmax'))
        modelCaptions.load_weights('./weights/cross-captions-weights.h5')

      
    if (weights == "cross-vecsi"):
        modelCaptionsScratch = Sequential()
        modelCaptionsScratch.add(Embedding(len(word_index_tokens)+1, dim, embeddings_initializer="uniform", input_length=max_sequence_length,trainable=True))
        modelCaptionsVecsiTokens = Sequential()
        modelCaptionsVecsiTokens.add(Embedding(len(word_index_tokens) + 1,dim, embeddings_initializer="uniform", input_length = max_sequence_length, trainable = False))
        modelCaptionsVecsiSyncons = Sequential()
        modelCaptionsVecsiSyncons.add(Embedding(len(word_index_syncons) + 1,dim, embeddings_initializer="uniform", input_length = max_sequence_length, trainable = False))
        modelMergeEmbeddings = Concatenate()([modelCaptionsScratch.output,modelCaptionsVecsiTokens.output,modelCaptionsVecsiSyncons.output])
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(5)(modelMergeEmbeddings)
        modelMergeEmbeddings = Conv1D(512, 5, activation="relu")(modelMergeEmbeddings)
        modelMergeEmbeddings = MaxPooling1D(35)(modelMergeEmbeddings)
        modelMergeEmbeddings = Reshape((1,1,512))(modelMergeEmbeddings)
        modelMergeEmbeddings = Flatten()(modelMergeEmbeddings)    
        modelMergeEmbeddings = Dense(128, activation='relu')(modelMergeEmbeddings)
        modelMergeEmbeddings = Dense(2, activation='softmax')(modelMergeEmbeddings)
        modelCaptions = Model([modelCaptionsScratch.input,modelCaptionsVecsiTokens.input,modelCaptionsVecsiSyncons.input], modelMergeEmbeddings)
        modelCaptions.load_weights('./weights/cross-vecsi-captions-weights.h5')

    return modelCaptions

def get_cat_figures_model (weights):
    modelImages = Sequential()
    modelImages.add(InputLayer(input_shape=(224,224,3)))
    modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(MaxPooling2D(2))
    modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(MaxPooling2D(2))
    modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(MaxPooling2D(2))
    modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
    modelImages.add(BatchNormalization())
    modelImages.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
    modelImages.add(BatchNormalization())
    modelImages.add(MaxPooling2D((28,28),2))
    modelImages.load_weights('./weights/'+weights+'-figures-weights.h5')      
      
    return modelImages

# Read the articles dataset that will be used to train and validate the model.
def tqa_data_extraction(dataset_folders, dataset_jsons, question_type):
    count = 0
    #Prepare data
    paragraphs = []
    figures_paragraphs = []
    questions = []
    figures_questions = []
    list_answers = [[],[],[],[]]
    correct_answers = []
    question_types = []
    split_cont = 0
    for dataset_json in dataset_jsons:
        folder = dataset_folders[split_cont]
        split_cont = split_cont+1
        with open(dataset_json, "r") as file:
            dataset = json.load(file)

        for doc in dataset:
            count=count+1
            sys.stdout.write("\r%d lessons processed" % count)
            sys.stdout.flush()

            question_list = [x for x in doc["questions"][question_type]]
            for question_id in question_list:
                n_answers = len (doc["questions"][question_type][question_id]["answerChoices"])
                
                if(question_type == "nonDiagramQuestions" and (n_answers!=4 or question_id == "NDQ_005923" or question_id == "NDQ_006171" or question_id == "NDQ_004046" or question_id =="NDQ_016510" or doc["questions"][question_type][question_id]["questionSubType"] != "Multiple Choice")):
                    continue

                #Questions
                tqa_questions_getter(folder, doc, question_id, question_type, questions,figures_questions)

                #Context
                tqa_context_getter(folder, doc, question_id, question_type, paragraphs, figures_paragraphs)

                #Answers
                tqa_answers_getter(doc, question_id, question_type, n_answers, list_answers)

                #Correct Answer (labeling)
                tqa_label_getter(doc, question_id, question_type, correct_answers)

    data_raw = [paragraphs,questions,list_answers[0],list_answers[1],list_answers[2],list_answers[3]]
    figures = [figures_paragraphs,figures_questions]
    print("\n")

    return data_raw, figures, correct_answers
    
def tqa_questions_getter(folder, doc, question_id, question_type, questions,figures_questions):
    question = doc["questions"][question_type][question_id]["beingAsked"]["processedText"]
    if(question_type == "diagramQuestions"):
      figure_path = folder+doc["questions"][question_type][question_id]["imagePath"]
      figure_file = open(figure_path, 'rb')
      figure = Image.open(figure_file)
      figure_resized = figure.resize((224,224), Image.ANTIALIAS)
      figure_array = np.array(figure_resized)
      figure.close()
      figure_file.close()
    if(question_type=="nonDiagramQuestions"):
      figure_array = np.zeros((224,224,3))
    figures_questions.append(figure_array)
    questions.append(question)
def tqa_context_getter(folder, doc, question_id, question_type, paragraphs, figures_paragraphs):
    question = doc["questions"][question_type][question_id]["beingAsked"]["processedText"]
    topics = [x for x in doc["topics"]]
    documents = []
    figs = []
    documents.append(question)
    figs.append("")
    for topic in topics:
      paragraph = doc["topics"][topic]["content"]["text"]
      figure_path = ""
      if len(doc["topics"][topic]["content"]["figures"])>0:
        figure_path = folder+doc["topics"][topic]["content"]["figures"][0]["imagePath"]
      figs.append(figure_path)
      documents.append(paragraph)
    tfidf = TfidfVectorizer().fit_transform(documents)
    pairwise_similarity = tfidf * tfidf.T
    score_max_index = np.argmax(pairwise_similarity[0,1:])+1
    score_max_paragraph = documents[score_max_index]
    score_max_figure = figs [score_max_index]
    score_max = pairwise_similarity[0,score_max_index]
    chosen_paragraph = score_max_paragraph
    chosen_figure = score_max_figure
    if chosen_figure == "":
        figure_array = np.zeros((224,224,3))
    else:
        figure_file = open(chosen_figure, 'rb')
        figure = Image.open(figure_file)
        figure_resized = figure.resize((224,224), Image.ANTIALIAS)
        figure_array = np.array(figure_resized)
        figure.close()
        figure_file.close()
    figures_paragraphs.append(figure_array)
    paragraphs.append(chosen_paragraph)
def tqa_answers_getter(doc, question_id,question_type,n_answers,list_answers):
    letter_list=["a","b","c","d"]
    for i in range(4):
        if(i < n_answers):
            letter = letter_list[i]
            answer = doc["questions"][question_type][question_id]["answerChoices"][letter]["processedText"]
        else:
            answer=""
        list_answers[i].append(answer)
def tqa_label_getter(doc, question_id, question_type, correct_answers):   
    correct_answer = doc["questions"][question_type][question_id] ["correctAnswer"]["processedText"]
    letter_list=["a","b","c","d"]
    correct_array = np.zeros(4)
    for i in range(4):
        if(letter_list[i]==correct_answer):
            correct_array[i]=1
    correct_answers.append(correct_array)

def tqa_data_refiner(data_raw,figures,correct_answers,tokenizer, weights, context_length, question_length):
    model = Sequential()
    model.add(InputLayer(input_shape=(224,224,3)))
    model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(2))
    model.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
    model.add(BatchNormalization())
    model.add(MaxPooling2D((28,28),2))
    model.load_weights('./weights/'+weights+'-figures-weights.h5')
    count = 0
    data =[]
    for i in range(len(data_raw)):
        if(i==0):
            max_len = context_length
        else:
            max_len= question_length
        sequences = tokenizer.texts_to_sequences(data_raw[i])
        data_refined = pad_sequences(sequences, maxlen=max_len, padding="post", truncating="post")
        data.append(data_refined)
        print("Shape of (" + str(count) + ") data tensor:" + str(data_refined.shape))
        if(i == 0):
            figure_feat = tqa_features_extraction(np.array(figures[i]),model)
            data.append(figure_feat)
            count = count+1
            print("Shape of (" + str(count) + ") data tensor:" + str(figure_feat.shape))
        if(i == 1):
            figure_feat = tqa_features_extraction(np.array(figures[i]),model)
            data.append(figure_feat)
            count = count+1
            print("Shape of (" + str(count) + ") data tensor:" + str(figure_feat.shape))
        count = count+1
    labels = np.array(correct_answers)
    print("Shape of labels tensor:", labels.shape)

    print("\n")

    return data, labels
def tqa_features_extraction(figure,model):
    features = model.predict(figure,batch_size=32,verbose=1)
    return features

def tqa_similarityMU(x):
    M = x[0]
    U = x[1]
    S = tf.matmul(M,U, transpose_b=True)
    res = tf.reduce_max(S,axis=2,keepdims=True)
    return res

def tqa_answerer(x):
    a = x[0]
    M = x[1]
    M_t = tf.transpose(M, [0, 2, 1])
    a_exp = tf.expand_dims(a, 1)
    m = tf.multiply(a_exp,M_t)
    res = tf.reduce_sum(m,axis=2,keepdims=True)
    return res

def tqa_similaritymC(x):
    m = x[0]
    C = x[1]
    C1, C2, C3, C4 = tf.split(C, 4, 2)
    C_list=[C1,C2,C3,C4]
    res_tmp=[]
    for C in C_list:
        C_t = tf.transpose(C, [0, 2, 1])
        C_sum = tf.reduce_sum(C_t, axis=2, keepdims=True)
        res_tmp.append(tf.matmul(m, C_sum,transpose_a = True))
    #res = tf.reduce_sum(tf.concat(res_tmp, 1),axis=2)
    res = tf.concat(res_tmp, 1)
    return res

def get_tqa_model(n_words, dim, dout, rdout, context_length, question_length):
    modelM = Sequential()
    modelM.add(InputLayer(input_shape=(context_length,),name="input_M"))
    modelM.add(Embedding(input_dim=n_words+1, output_dim=dim, input_length=context_length, embeddings_initializer="uniform", trainable=True,name="embedding_M"))
    modelM.add(LSTM(units=dim, return_sequences=True, name="lstm_M", dropout=dout, recurrent_dropout=rdout))

    modelMF = Sequential()
    modelMF.add(InputLayer(input_shape=(1,1,512,),name="input_MF"))
    modelMF.add(Reshape((1,512,),name="reshape_MF"))
    modelMF.add(Dense(256, activation="tanh",name="perceptron_MF_1"))
    modelMF.add(Dense(dim, activation="tanh",name="perceptron_MF_2"))

    modelInMMF = Concatenate(name = "concatenateMMF", axis=1)([modelM.output,modelMF.output])
    modelMMF = Model([modelM.input,modelMF.input], modelInMMF)

    modelU = Sequential()
    modelU.add(InputLayer(input_shape=(question_length,),name="input_U"))
    modelU.add(Embedding(input_dim=n_words+1, output_dim=dim, input_length=question_length, embeddings_initializer="uniform", trainable=True,name="embedding_U"))
    modelU.add(LSTM(units=dim, return_sequences=True, name="lstm_U", dropout=dout, recurrent_dropout=rdout))

    modelUF = Sequential()
    modelUF.add(InputLayer(input_shape=(1,1,512,),name="input_UF"))
    modelUF.add(Reshape((1,512,),name="reshape_UF"))
    modelUF.add(Dense(256, activation="tanh",name="perceptron_UF_1"))
    modelUF.add(Dense(dim, activation="tanh",name="perceptron_UF_2"))

    modelInUUF = Concatenate(name = "concatenateUUF", axis=1)([modelU.output,modelUF.output])
    modelUUF = Model([modelU.input,modelUF.input], modelInUUF)

    modelC1 = Sequential()
    modelC1.add(InputLayer(input_shape=(question_length,),name="input_C1"))
    modelC1.add(Embedding(input_dim=n_words+1, output_dim=dim, input_length=question_length, embeddings_initializer="uniform", trainable=True,name="embedding_C1"))
    modelC1.add(LSTM(units=dim, return_sequences=True, name="lstm_C1", dropout=dout, recurrent_dropout=rdout))
    modelC2 = Sequential()
    modelC2.add(InputLayer(input_shape=(question_length,),name="input_C2"))
    modelC2.add(Embedding(input_dim=n_words+1, output_dim=dim, input_length=question_length, embeddings_initializer="uniform", trainable=True,name="embedding_C2"))
    modelC2.add(LSTM(units=dim, return_sequences=True, name="lstm_C2", dropout=dout, recurrent_dropout=rdout))
    modelC3 = Sequential()
    modelC3.add(InputLayer(input_shape=(question_length,),name="input_C3"))
    modelC3.add(Embedding(input_dim=n_words+1, output_dim=dim, input_length=question_length, embeddings_initializer="uniform", trainable=True,name="embedding_C3"))
    modelC3.add(LSTM(units=dim, return_sequences=True, name="lstm_C3", dropout=dout, recurrent_dropout=rdout))
    modelC4 = Sequential()
    modelC4.add(InputLayer(input_shape=(question_length,),name="input_C4"))
    modelC4.add(Embedding(input_dim=n_words+1, output_dim=dim, input_length=question_length, embeddings_initializer="uniform", trainable=True,name="embedding_C4"))
    modelC4.add(LSTM(units=dim, return_sequences=True, name="lstm_C4", dropout=dout, recurrent_dropout=rdout))

    modelInC = Concatenate(name = "concatenate")([modelC1.output,modelC2.output,modelC3.output,modelC4.output])
    modelC = Model([modelC1.input,modelC2.input,modelC3.input,modelC4.input], modelInC)

    modelIna = Lambda(tqa_similarityMU, output_shape=(context_length+1,),name="similarityMU")([modelMMF.output, modelUUF.output])
    modelIna = Dense(context_length+1,activation="softmax",name="softmax_a")(modelIna)
    modela = Model([modelM.input,modelMF.input,modelU.input,modelUF.input], modelIna)

    modelInm = Lambda(tqa_answerer, output_shape=(dim,),name="answerer") ([modela.output, modelMMF.output])
    modelm = Model([modelM.input,modelMF.input,modelU.input,modelUF.input], modelInm)

    modelIn = Lambda(tqa_similaritymC, output_shape=(4,),name="similaritymC")([modelm.output,modelC.output])
    modelIn = Dense(4, activation="softmax",name="softmax_y") (modelIn)
    model = Model([modelM.input,modelMF.input,modelU.input,modelUF.input,modelC1.input,modelC2.input,modelC3.input,modelC4.input], modelIn)
    
    return model

def get_dropouts(question_type):
    if (question_type == "nonDiagramQuestions"):
        dout = 0.5
        rdout = 0.5
    if (question_type == "diagramQuestions"):
        dout = 0.0
        rdout = 0.0
    return dout,rdout
        
def get_tqa_sequence_lengths(question_type):
    if (question_type == "nonDiagramQuestions"):
        context_length = 630
        question_length = 73
    if (question_type == "diagramQuestions"):
        context_length = 418
        question_length = 46
    return context_length, question_length