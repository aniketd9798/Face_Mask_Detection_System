import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# 1. Set up initial parameters
INIT_LR = 1e-4       # Learning rate (how fast the AI learns)
EPOCHS = 20          # How many times the AI loops through the whole dataset
BS = 32              # Batch size (how many images to look at once)
DIRECTORY = "dataset" # The folder you created in Step 1

print("[INFO] Loading images...")

# 2. Data Augmentation (This creates slightly modified versions of your images to help the AI learn better)
aug = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Load the images from the directories
train_generator = aug.flow_from_directory(
    DIRECTORY,
    target_size=(224, 224),
    batch_size=BS,
    class_mode="categorical"
)

# 3. Load MobileNetV2 (The base brain)
print("[INFO] Compiling model...")
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

# Construct the "head" of the model (This customizes it specifically for Mask / No Mask)
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 2 outputs: Mask or No Mask

# Combine the base and the head
model = Model(inputs=baseModel.input, outputs=headModel)

# Freeze the base layers so we don't destroy their pre-trained knowledge
for layer in baseModel.layers:
    layer.trainable = False

# 4. Compile and Train
opt = Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

print("[INFO] Training head...")
H = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator) // BS,
    epochs=EPOCHS
)

# 5. Save the trained model to your computer!
print("[INFO] Saving mask detector model...")
model.save("mask_detector.h5")
print("[INFO] Done! You now have a trained AI brain.")