# Convolutional Neural Network
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

# Image size
img_width, img_height = 128, 128

def create_classifier(p, input_shape=(64, 64, 3)):
    # Intialising the CNN
    classifier = Sequential()
    # Convolution and Pooling / Max Pooling layer
    classifier.add(Conv2D(32, (3, 3), input_shape=input_shape, activation='relu' ))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # Convolution and Pooling / Max Pooling layer
    classifier.add(Conv2D(32, (3, 3), activation='relu' ))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    classifier.add(Conv2D(64, (3, 3), activation='relu' ))
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flattening
    classifier.add(Flatten())
    # Full connection
    classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dropout(p))
    # classifier.add(Dense(units=64, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid'))

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

# Fitting the CCN to the images
def fit_run_classifier(bs = 32, epochs=10):
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                    target_size=(img_width, img_height),
                                                    batch_size=bs,
                                                    class_mode='binary')

    test_set = test_datagen.flow_from_directory('dataset/test_set',
                                                target_size=(img_width, img_height),
                                                batch_size=bs,
                                                class_mode='binary')
    classifier = create_classifier(p=0.5, input_shape=(img_width, img_height, 3))
    classifier.fit_generator(training_set,
                            epochs=10,
                            validation_data=test_set)

def main():
    fit_run_classifier(bs=32, epochs=12)

if __name__ == "__main__":
    main()

# # Save model/clissifier - Uncomment this
# classifier.save('dog-or-cat.h5')

# Import model/classifier
classifier = load_model('dog-or-cat.h5') 

# Making prediction
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_3.jpg', target_size=(img_width, img_height))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
test_image = test_image/255.0
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'

#loss: 0.4136 - accuracy: 0.8040 - val_loss: 0.7613 - val_accuracy: 0.7835
#loss: 0.3839 - accuracy: 0.8276 - val_loss: 0.4968 - val_accuracy: 0.7945