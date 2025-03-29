import tensorflow as tf

model_paths = ["trained_mobilenet_model.keras", "trained_model.h5", "trained_model.keras"]

for path in model_paths:
    try:
        model = tf.keras.models.load_model(path)
        print(f"✅ Successfully loaded: {path}")
        print(model.summary())  # Check architecture
    except Exception as e:
        print(f"❌ Failed to load {path}: {e}")
