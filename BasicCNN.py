import os
import json
from Utils import generate_image_data, model_evaluator, plot_accuracy_loss, specie_error_rate_evaluator, plot_species, save_trained_model, plot_confusion_matrix
from cnnModels import basic_cnn

model = basic_cnn()

base_directory = os.getcwd()
dataset_directory = os.path.join(base_directory, 'Dataset')
training_directory = os.path.join(dataset_directory, 'training')
validation_directory = os.path.join(dataset_directory, 'validation')
test_directory = os.path.join(dataset_directory, 'testing')

# Get batches of data
train_generator = generate_image_data(training_directory)
validation_generator = generate_image_data(validation_directory)

history = model.fit_generator(train_generator,
                              steps_per_epoch = 2,
                              epochs = 1,
                              validation_data = validation_generator,
                              validation_steps = 180)

with open('basic_cnn.json', 'w') as f:
    json.dump(history.history, f)
predict = model_evaluator(model=model)
print('Testing accuracy: ', predict[1])

# Plot accuracy and loss
plot_accuracy_loss(history)

test_generator = generate_image_data(test_directory, shuffle = False, batch_size = 1)
error_specie = specie_error_rate_evaluator(model)

# Plot individual expression error rate
plot_species(error_specie, 'Individual expression error rate (Overall %.2f%% accuracy)' % (predict[1] * 100))

#Plot confusion matrix
plot_confusion_matrix(model)

save_trained_model(model, 'basic_cnn.h5')

