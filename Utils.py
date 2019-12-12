import os
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from keras import models
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.externals import joblib
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sb


base_directory = os.getcwd()
dataset_directory = os.path.join(base_directory, 'Dataset')
training_directory = os.path.join(dataset_directory, 'training')
validation_directory = os.path.join(dataset_directory, 'validation')
test_directory = os.path.join(dataset_directory, 'testing')

Models = os.path.join(os.getcwd(), 'Models')

digitsToSpecies = {'0' : 'alouatta_palliata', 
                   '1' : 'erythrocebus_patas', 
                   '2' : 'cacajao_calvus', 
                   '3' : 'macaca_fuscata',
                   '4' : 'cebuella_pygmea', 
                   '5' : 'cebus_capucinus', 
                   '6' : 'mico_argentatus'
                  }


def plot_confusion_matrix(model):

    test_generator = generate_image_data(test_directory, shuffle=False, batch_size=20)
    Y_pred = model.predict_generator(test_generator, 180)  #180 - 2nd param
    y_pred = np.argmax(Y_pred, axis=1)
    print('*** Confusion Matrix ***')
    confusion_matrix_result = confusion_matrix(test_generator.classes, y_pred)
    print(confusion_matrix_result)
    class_names = ['alouatta_palliata', 'erythrocebus_patas', 'cacajao_calvus', 'macaca_fuscata', 'cebuella_pygmea', 
                    'cebus_capucinus', 'mico_argentatus']
    df_cm = pd.DataFrame(confusion_matrix_result, index=[i for i in class_names],
                         columns=[i for i in class_names])
    plt.figure(figsize=(10, 7))     
    sb.heatmap(df_cm, annot=True)
    plt.show()

    print('*** Classification Report ***')
    classified_report = classification_report(test_generator.classes, y_pred, class_names=class_names)
    print(classified_report)


def pixel2image(pixels, dst_dir, fname, mode='L'):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    img_path = os.path.join(dst_dir, fname)
    im = Image.fromarray(pixels).convert(mode)
    im.save(img_path)


def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points


def plot_accuracy_loss(history):
    accuracy = smooth_curve(history.history['acc'])
    validation_accuracy = smooth_curve(history.history['val_acc'])
    loss = smooth_curve(history.history['loss'])
    validation_loss = smooth_curve(history.history['val_loss'])

    epochs = range(1, len(accuracy) + 1)
    sb.set()
    plt.plot(epochs, accuracy, 'bo', label='Training accuracy')
    plt.plot(epochs, validation_accuracy, 'b', label='Validation accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.figure()
    sb.set()

    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, validation_loss, 'b', label='Validation loss')
    plt.title('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()


def generate_image_data(data_dir,
                         data_augment=False,
                         batch_size=20,
                         target_size=(224, 224),  
                         #color_mode='grayscale',
                         class_mode='categorical',
                         shuffle=True):


    data_generator = ImageDataGenerator(rescale=1./255)

    generator = data_generator.flow_from_directory(data_dir,
                                            target_size=target_size,
                                            #color_mode=color_mode,
                                            batch_size=batch_size,
                                            shuffle=shuffle,
                                            class_mode=class_mode)
    return generator


def model_evaluator(model=None, file_path=None):

    if not model:
        assert(file_path)
        model = models.load_model(file_path)
    test_generator = generate_image_data(test_directory, batch_size=1, shuffle=False)

    nb_samples = len(test_generator)
    evaluation_score = model.evaluate_generator(test_generator, steps=nb_samples)

    return evaluation_score


def assign_probability_to_class(y_pred):

    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        return np.argmax(y_pred, axis=1)
    return np.array([1 if p > 0.5 else 0 for p in y_pred])


def specie_error_rate_evaluator(model):

    batch_size = 1
    test_generator = generate_image_data(test_directory, batch_size = batch_size)
    sample_count = len(test_generator)
    Y_test = np.zeros(shape = sample_count)
    Y_pred = np.zeros(shape = sample_count)

    i = 0
    for X, labels_batch in test_generator:
        pred = model.predict(X)
        Y_pred[i] = assign_probability_to_class(pred)
        Y_test[i * batch_size: (i + 1) * batch_size] = np.argmax(labels_batch, axis = 1)
        i += 1
        if i * batch_size >= sample_count:
            break

    classes = len(set(Y_test))
    x_num, y_num = [0] * classes, [0] * classes
    for pred, test in zip(Y_pred, Y_test):
        y_num[int(test)] += 1
        if pred != test:
            x_num[int(test)] += 1

    err = [i/j for i, j in zip(x_num, y_num)]
    return err


def specie_error_rate_evaluator_svm(clf, test_features, test_labels):
    predict_specie = clf.predict(test_features)
    classes = len(set(test_labels))
    class_total, error_labels = [0] * classes, [0] * classes
    for predict_label, test_label in zip(predict_specie, test_labels):
        class_total[int(test_label)] += 1
        if predict_label != test_label:
            error_labels[int(test_label)] += 1

    return [error/total for error, total in zip(error_labels, class_total)]


def plot_species(err, title):
    s = pd.Series(
        err,
        index=['alouatta_palliata', 'erythrocebus_patas', 'cacajao_calvus', 'macaca_fuscata', 'cebuella_pygmea', 
               'cebus_capucinus', 'mico_argentatus']
    )

    sb.set()
    plt.title(title)
    plt.ylabel('error rate')
    plt.xlabel('species')

    graph_colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c'] 

    s.plot(
        kind='bar',
        color=graph_colors,
    )
    plt.show()


def save_trained_model(model, file_name):
    file_path = os.path.join(Models, file_name)
    if file_name.endswith('h5'):
        model.save(file_path)
    elif file_name.endswith('json'):
        model.to_json(file_path)


def load_trained_model(file_name):
    Model = os.path.join(os.getcwd(), 'Models')
    file_path = os.path.join(Model, file_name)
    model = None
    if file_name.endswith('h5'):
        model = models.load_model(file_path)
    elif file_name.endswith('hdf5'):
        model = models.load_model(file_path)
    elif file_name.endswith('json'):
        with open('model.json', 'r') as json_file:
            json_model = json_file.read()
            model = models.model_from_json(json_model)
    elif file_name.endswith('pkl'):
        joblib.load(file_path)
    assert(model)
    return model


def svm_feature_extractor(directory, model, layer_name=None,
                             batch_size=20):

    intermediate_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    generator = generate_image_data(directory, batch_size=batch_size)
    sample_count = len(generate_image_data(directory, batch_size=1))
    print(os.path.split(directory)[-1], 'Dataset: ', sample_count)

    features = np.zeros(shape = (sample_count, 1024))  #check

    labels = np.zeros(shape = (sample_count))

    i = 0
    for inputs_batch, labels_batch in generator:
        intermediate_output = intermediate_model.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = intermediate_output
        labels[i * batch_size: (i + 1) * batch_size] = np.argmax(labels_batch, axis=1)
        i += 1
        if i * batch_size >= sample_count:
            break
    np.reshape(features, (sample_count, 1024)) #check
    return features, labels


def svc(training_data, training_label, test_data, test_label):
    model_file = "cnn_and_svm_linear.joblib"
    print("Training SVM with rbf kernel funtion...")
    classifier = SVC(C=1.0, kernel="linear", cache_size=3000)
    classifier.fit(training_data, training_label)

    joblib.dump(classifier, model_file)
    print('dumped')
    pred_testlabel = classifier.predict(test_data)
    num = len(pred_testlabel)
    accuracy = len([1 for i in range(num) if test_label[i] == pred_testlabel[i]]) / float(num)

    return classifier, accuracy
