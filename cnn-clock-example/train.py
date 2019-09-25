from keras import backend as keras_backend
from representation import *
import models
import h5py

keras_backend.set_image_dim_ordering('th')

train_dir = os.path.join('data', 'train')
x_train, y_train = get_images(train_dir)

validation_dir = os.path.join('data', 'validation')
x_validation, y_validation = get_images(validation_dir)

num_train_samples = len(x_train)
num_validation_samples = len(x_validation)

image_width = image_height = 32

num_color_channels = 1

print('num_train_samples', num_train_samples)
print('num_validation_samples', num_validation_samples)

model = models.get_cnn_model(num_color_channels, image_width, image_height)
print(model.summary())

model.fit(x_train, y_train, batch_size=32, epochs=300)

model.save('clock.h5')
