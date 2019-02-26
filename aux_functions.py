from keras.models import Model, Sequential
from keras.layers import Embedding, Conv1D, Conv2D, MaxPooling1D, MaxPooling2D, Flatten, Dense, Concatenate, InputLayer, Reshape, BatchNormalization

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

