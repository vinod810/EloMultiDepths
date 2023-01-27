import os
import sys
import tensorflow as tf
from tensorflow.python.keras import models
from tensorflow.python.layers import layers

from pgn_to_data import Max_CP_Loss, Analysis_Moves, Analysis_List, Train_Val_Ratio, Max_Elo, Min_Elo
from pgn_to_data import Train_Data_File, Val_Data_File, Train_Label_File, Val_Label_File

Batch_Size = 32
EPOCHS = 10
AUTO = tf.data.experimental.AUTOTUNE
Data_Record_Size = Analysis_Moves * len(Analysis_List)


def read_data(tf_bytestring):
    loss = tf.io.decode_raw(tf_bytestring, tf.int32)
    loss = tf.cast(loss, tf.float32) / Max_CP_Loss  # normalize 0.0 .. 1.0
    loss = tf.reshape(loss, [Data_Record_Size])
    return loss


def read_label(tf_bytestring):
    elo = tf.io.decode_raw(tf_bytestring, tf.int32)
    elo = (tf.cast(elo, tf.float32) - Min_Elo) / (Max_Elo - Min_Elo)  # normalize 0.0 .. 1.0
    return elo


def load_dataset(data_file, label_file):
    data_dataset = tf.data.FixedLengthRecordDataset(filenames=[data_file],
                                                    record_bytes=4 * Data_Record_Size,
                                                    header_bytes=0, footer_bytes=0)
    data_dataset = data_dataset.map(read_data, num_parallel_calls=16)

    label_dataset = tf.data.FixedLengthRecordDataset(filenames=[label_file],
                                                     record_bytes=4, header_bytes=0, footer_bytes=0)
    label_dataset = label_dataset.map(read_label, num_parallel_calls=16)

    dataset = tf.data.Dataset.zip((data_dataset, label_dataset))

    # for data in dataset:
    #    print(data)
    # exit()
    return dataset


def make_training_dataset(data_file, label_file):
    dataset = load_dataset(data_file, label_file)
    dataset = dataset.shuffle(5000, reshuffle_each_iteration=True)
    # dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.batch(Batch_Size, drop_remainder=True)  # important on TPU, batch size must be fixed
    dataset = dataset.prefetch(AUTO)  # fetch next batches while training on the current one
    return dataset


def make_validation_dataset(data_file, label_file):
    dataset = load_dataset(data_file, label_file)
    dataset = dataset.batch(Batch_Size, drop_remainder=True)
    # dataset = dataset.repeat()  # Mandatory for Keras for now
    dataset = dataset.prefetch(AUTO)
    return dataset


def main(model_dir, data_dir, total_items_count):
    steps_per_epoch = int(total_items_count * (Train_Val_Ratio - 1) / Train_Val_Ratio / Batch_Size)

    data_sub_dir = data_dir + '/' + str(total_items_count)
    train_model(model_dir, data_sub_dir, steps_per_epoch)


def train_model(model_dir, data_sub_dir, steps_per_epoch):
    training_dataset = make_training_dataset(data_sub_dir + '/' + Train_Data_File,
                                             data_sub_dir + '/' + Train_Label_File)
    validation_dataset = make_validation_dataset(data_sub_dir + '/' + Val_Data_File,
                                                 data_sub_dir + '/' + Val_Label_File)

    # model = tf.keras.Sequential([
    #     tf.keras.layers.Flatten(input_shape=[Data_Record_Size, 1]),
    #     tf.keras.layers.Dense(200, activation="relu"),
    #     tf.keras.layers.Dense(100, activation="relu"),
    #     tf.keras.layers.Dense(Data_Record_Size // Analysis_Moves, activation="relu"),
    #     tf.keras.layers.Dense(1, activation='sigmoid'),  # 'linear')
    # ])
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(32, (len(Analysis_List)), strides=len(Analysis_List),
                               input_shape=(Data_Record_Size, 1)),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        # model.add(layers.MaxPooling2D((2, 2)))
        tf.keras.layers.Conv1D(32, (Analysis_Moves), strides=Analysis_Moves),
        tf.keras.layers.BatchNormalization(center=True, scale=False),
        tf.keras.layers.Activation('relu'),

        # model.add(layers.MaxPooling2D((2, 2)))
        # model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        tf.keras.layers.Flatten(),
        tf.keras.layers.BatchNormalization(scale=False, center=True),
        tf.keras.layers.Dense(32, activation='relu'),

        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid'),
    ])
    model.summary()

    model.compile(optimizer='adam', loss='mse', metrics=['mse'])

    # Train the model
    checkpoint_filepath = model_dir + '/checkpoint'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=2,
        verbose=0,
        mode='auto',
        baseline=None,
        restore_best_weights=True,
        start_from_epoch=0
    )

    # steps_per_epoch = int(items_count * Train_Data_Factor) // BATCH_SIZE
    print("Steps per epoch: ", steps_per_epoch)
    try:
        # model.fit(training_dataset, steps_per_epoch=steps_per_epoch, epochs=EPOCHS,
        model.fit(training_dataset, epochs=EPOCHS,
                  validation_data=validation_dataset, validation_steps=1,
                  callbacks=[model_checkpoint_callback, early_stopping_callback])
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received!")
        exit()

    model.load_weights(checkpoint_filepath)
    if model_dir is not None:
        print('Saving model ...')
        os.makedirs(model_dir, exist_ok=True)
        model.save(model_dir)
    else:
        print('Model not saved')
    '''
    for data in validation_dataset:
        predictions = model.predict(data[0])
        for i in range(len(predictions)):
            print(data[1][i] * 1000 + 1000, predictions[i] * 1000 + 1000)
        exit()
    '''

# export TF_CPP_MIN_LOG_LEVEL=2
if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python3 model.py model-dir data-dir items-count")
        exit()

    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
