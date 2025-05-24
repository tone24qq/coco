from tensorflow.keras import layers, models, Input

def build_model(input_shape=(10, 12, 4)):
    inp = Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    mask_out = layers.Conv2D(1, (1, 1), activation='sigmoid', name="mask_out")(x)

    x_flat = layers.Flatten()(x)
    value_out = layers.Dense(121, activation='softmax', name="value_out")(x_flat)

    model = models.Model(inputs=inp, outputs=[mask_out, value_out])
    return model
