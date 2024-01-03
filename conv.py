import tensorflow as tf
import matplotlib.pyplot as plt
import math

# Load the model from .model file
model = tf.keras.models.load_model('ResNet50V2_Dense512_KFold10_TA_60.model')
model.summary()

num_layers = len(model.layers)
print("Number of layers in the model:", num_layers)

# Choose the layers for visualization
layers_to_visualize = [10, 100, 150]

# Define a function to visualize convolutional outputs
def visualize_conv_outputs(layer_number):
    # Create a sub-model that outputs the desired layer's activations
    sub_model = tf.keras.models.Model(inputs=model.input, outputs=model.layers[layer_number].output)

    # Load and preprocess an example image for visualization
    # example_image = tf.keras.preprocessing.image.load_img('Tr-no_0026.jpg', target_size=(224, 224))
    example_image = tf.keras.preprocessing.image.load_img('Tr-gl_0315.jpg', target_size=(224, 224))
    # example_image = tf.keras.preprocessing.image.load_img('Tr-me_0019.jpg', target_size=(224, 224))
    # example_image = tf.keras.preprocessing.image.load_img('Tr-pi_0877.jpg', target_size=(224, 224))
    example_image = tf.keras.preprocessing.image.img_to_array(example_image)
    example_image = tf.keras.applications.resnet_v2.preprocess_input(example_image)
    example_image = tf.expand_dims(example_image, axis=0)

    # Get the activations for the example image
    activations = sub_model.predict(example_image)

    # Print the number of feature maps
    num_feature_maps = activations.shape[-1]
    print("Number of feature maps for layer", layer_number, ":", num_feature_maps)

    # Plot all feature maps
    rows = int(math.sqrt(num_feature_maps))
    cols = math.ceil(num_feature_maps / rows)

    plt.figure(figsize=(10, 10))
    for i in range(num_feature_maps):
        feature_map = activations[0, :, :, i]
        plt.subplot(rows, cols, i + 1)
        plt.imshow(feature_map.squeeze(), cmap='gray')
        plt.axis('off')
    plt.show()

# Visualize the convolutional outputs for the chosen layers
for layer in layers_to_visualize:
    visualize_conv_outputs(layer)
