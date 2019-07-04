import gc
import os

import pandas

from keras.layers import Dense, Conv2D, Activation, MaxPooling2D, Dropout, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping


def main():
    gc.collect()

    data_dir = os.path.abspath('./dataset')
    index_df = pandas.read_csv(os.path.join(data_dir, 'index.csv'))

    train = index_df.iloc[:round(len(index_df) * .9)]
    test = index_df.iloc[round(len(index_df) * .9):]

    train_datagen = ImageDataGenerator(rescale=1. / 255., validation_split=0.25)
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=data_dir,
        x_col="file_path",
        y_col="emotion",
        batch_size=32,
        seed=42,
        shuffle=True,
        subset="training",
        class_mode="categorical",
        target_size=(64, 64))

    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=train,
        directory=data_dir,
        x_col="file_path",
        y_col="emotion",
        batch_size=32,
        seed=42,
        shuffle=True,
        subset="validation",
        class_mode="categorical",
        target_size=(64, 64))

    gc.collect()

    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))

    model.add(Activation('hard_sigmoid'))
    model.add(Dropout(0.5))

    model.add(Dense(8, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    gc.collect()

    step_size_generator = train_generator.n // train_generator.batch_size
    step_size_validate = validation_generator.n // validation_generator.batch_size

    model.fit_generator(
        generator=train_generator,
        validation_data=validation_generator,
        steps_per_epoch=step_size_generator,
        validation_steps=step_size_validate,
        epochs=100,
        use_multiprocessing=True,
        callbacks=[EarlyStopping(monitor='loss', mode='min', verbose=1, patience=5)]
    )

    gc.collect()

    print('Model Done Training!')

    model.save('model.h5')

    print('Model Saved!')

    print('Evaluating model...')

    test_datagen = ImageDataGenerator(rescale=1. / 255.)
    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test,
        directory=data_dir,
        x_col="file_path",
        y_col="emotion",
        batch_size=32,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(64, 64))

    test_steps = test_generator.n / test_generator.batch_size
    scores = model.evaluate_generator(generator=test_generator, steps=test_steps, verbose=1, use_multiprocessing=True)

    print("Model accuracy %.2f%%" % (scores[1] * 100))

    del model

    gc.collect()


if __name__ == "__main__":
    main()
