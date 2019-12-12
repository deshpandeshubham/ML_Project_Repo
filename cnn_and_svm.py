import Utils
from Utils import load_trained_model, svm_feature_extractor, svc, specie_error_rate_evaluator_svm, save_trained_model
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sn


#load the previously trained deep CNN model
h5filename = 'dense_cnn.h5'
model = load_trained_model(h5filename)
model.summary()

#Extract features from last dense layer of CNN model
layer_name = 'dense_1'
train_features, train_labels = svm_feature_extractor(Utils.training_directory, model, layer_name=layer_name)
validation_features, validation_labels = svm_feature_extractor(Utils.validation_directory, model,
                                                                  layer_name=layer_name)
test_features, test_labels = svm_feature_extractor(Utils.test_directory, model, layer_name=layer_name)

#Train SVM using the extracted features
classifier, accuracy = svc(train_features, train_labels, validation_features, validation_labels)
print("Cnn and Svm (Accuracy %.2f%% )" % (accuracy * 100))

#Test SVM
evaluation_score = classifier.score(test_features, test_labels)
print(evaluation_score)
err = specie_error_rate_evaluator_svm(classifier, test_features, test_labels)
Utils.plot_species(err, 'Individual expression error rate')
Y_pred = y_pred = classifier.predict(test_features)

#Confusion matrix
print('*** Confusion Matrix ***')
confusion_matrix_result = confusion_matrix(test_labels, y_pred)
print(confusion_matrix_result)
class_names = ['alouatta_palliata', 'erythrocebus_patas', 'cacajao_calvus', 'macaca_fuscata', 'cebuella_pygmea', 
               'cebus_capucinus', 'mico_argentatus']
df_cm = pd.DataFrame(confusion_matrix_result, index=[i for i in class_names],
                     columns=[i for i in class_names])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
plt.show()

#Classification report
print('*** Classification Report ***')
classification_report = classification_report(test_labels, y_pred, target_names=class_names)
print(classification_report)

save_trained_model(model, 'cnn_and_svm.h5')
