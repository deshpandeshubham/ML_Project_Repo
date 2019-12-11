import os
import json
from Utils import image_data_generator, evaluate_model, plt_acc_loss,  \
    evaluate_expression_error_rate, plt_expression, save_model, load_model, plt_confusion_matrix
from cnnModels import basic_cnn, dense_cnn, Lenet


model = basic_cnn()

base_dir = os.getcwd()
dataset_dir = os.path.join(base_dir, 'Dataset')
train_dir = os.path.join(dataset_dir, 'training')
validation_dir = os.path.join(dataset_dir, 'validation')
test_dir = os.path.join(dataset_dir, 'testing')

# Get batches of data
train_generator = image_data_generator(train_dir)
validation_generator = image_data_generator(validation_dir)

history = model.fit_generator(train_generator,
                              steps_per_epoch = 2,
                              epochs = 1,
                              validation_data = validation_generator,
                              validation_steps = 180)

with open('basic_cnn.json', 'w') as f:
    json.dump(history.history, f)
predict = evaluate_model(model=model)
print('Testing accuracy: ', predict[1])

# Plot accuracy and loss
plt_acc_loss(history)

test_generator = image_data_generator(test_dir, shuffle = False, batch_size = 1)
err_expression = evaluate_expression_error_rate(model)

# Plot individual expression error rate
plt_expression(err_expression, 'Individual expression error rate (Overall %.2f%% accuracy)' % (predict[1] * 100))

#Plot confusion matrix
#plt_confusion_matrix(model)

save_model(model, 'basic_cnn.h5')

