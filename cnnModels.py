from keras import layers
from keras.layers.normalization import BatchNormalization
from keras import optimizers
from keras import models
import Utils
from Utils import load_trained_model, svm_feature_extractor,svc,specie_error_rate_evaluator_svm


def basic_cnn():
    print('Basic CNN')    
    model = models.Sequential()
    print('Basic CNN 1')
    model.add(layers.Conv2D(32, (3, 3), input_shape=(48, 48, 1), activation='relu'))
    print('Basic CNN 2')
    model.add(layers.MaxPooling2D((2, 2)))
    print('Basic CNN 3')
    model.add(layers.Flatten())
    print('Basic CNN 4')
    model.add(layers.Dropout(0.5))
    print('Basic CNN 5')
    model.add(layers.Dense(1024, activation='relu'))
    print('Basic CNN 6')
    model.add(layers.Dense(7, activation='softmax'))
    print('Basic CNN 7')
    model.summary()
    print('Basic CNN 8')
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
                  metrics=['acc'])
    return model

def dense_cnn():
    # conv  block 1
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))    
    model.add(BatchNormalization(axis=-1))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # conv  block 2
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    # conv  block 3
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization(axis=-1))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
                  metrics=['accuracy'])
    return model


def Lenet():
    # conv  block 1
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1024, activation='relu'))
    model.add(layers.Dense(7, activation='softmax'))
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4, decay=1e-6),
                  metrics=['accuracy'])
    return model



def cnn_and_svm():
    #Load the dense model
    h5filename = 'dense_cnn.h5'
    model = load_trained_model(h5filename)
    model.summary()
    layer_name = 'dense_1'
    train_features, train_labels = svm_feature_extractor(Utils.training_directory, model, layer_name=layer_name)
    validation_features, validation_labels = svm_feature_extractor(Utils.validation_directory, model,
                                                                      layer_name=layer_name)
    test_features, test_labels = svm_feature_extractor(Utils.test_directory, model, layer_name=layer_name)
    model_file = "cnn_and_svm.joblib"
    classifier, accuracy = svc(train_features, train_labels, validation_features, validation_labels)
    print("Cnn and Svm (Accuracy %.2f%% )" % (accuracy * 100))
    Utils.save_trained_model(model, 'cnn_and_svm.h5')
    #classifier = joblib.load(model_file)
    evaluation_score = classifier.evaluation_score(test_features, test_labels)
    print(evaluation_score)
    err= specie_error_rate_evaluator_svm(classifier,test_features, test_labels)
    Utils.plot_species(err, 'Individual species error rate')
    return classifier