import torch
import torch.nn.functional as F
from torchvision import models, transforms
import numpy as np
import cv2
import imageio
import os

def generate_individual_heatmap(model, image, save_dir, image_name):
    # Set the model in evaluation mode
    model.eval()

    # Define hooks to capture activations and gradients
    activation = {}
    def hook_fn(module, input, output):
        activation['value'] = output.detach()
    hook = model.register_forward_hook(hook_fn)

    # Forward pass
    input_image = image.unsqueeze(0)  # Add batch dimension
    output = model(input_image)

    # Get the predicted class index
    pred_index = output.argmax(dim=1).item()

    # Calculate the gradient of the predicted class output w.r.t. the activations
    model.zero_grad()
    target = torch.FloatTensor([pred_index])
    output.backward(gradient=target)

    # Get the activations and gradients
    activations = activation['value'].squeeze().cpu().numpy()
    gradients = model.features[-1].weight.grad.squeeze().cpu().numpy()

    # Calculate the importance weights
    importance_weights = np.mean(gradients, axis=(1, 2))

    # Generate the heatmap
    heatmap = np.dot(activations, importance_weights.reshape(1, -1))
    heatmap = np.maximum(heatmap, 0)  # ReLU operation
    heatmap /= np.max(heatmap)  # Normalize

    # Resize the heatmap to match the input image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))

    # Apply colormap to the heatmap
    heatmap_colormap = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

    # Overlay the heatmap on the input image
    heatmap_on_image = cv2.addWeighted(np.uint8(255 * image.permute(1, 2, 0)), 0.5,
                                        heatmap_colormap, 0.5, 0)

    # Save the heatmap image
    heatmap_path = os.path.join(save_dir, f'{image_name}_heatmap.png')
    imageio.imwrite(heatmap_path, heatmap_on_image)

    # Remove the hook
    hook.remove()

    # Also save the heatmap separately
    heatmap_only_path = os.path.join(save_dir, f'{image_name}_heatmap_only.png')
    imageio.imwrite(heatmap_only_path, heatmap_colormap)

def combine_heatmaps(heatmap_paths, save_dir, combined_heatmap_name='combined_heatmap.png'):
    # Read and combine heatmaps
    heatmaps = [imageio.imread(path) for path in heatmap_paths]
    combined_heatmap = np.hstack(heatmaps)

    # Save the combined heatmap image
    combined_heatmap_path = os.path.join(save_dir, combined_heatmap_name)
    imageio.imwrite(combined_heatmap_path, combined_heatmap)

def generate_and_combine_heatmaps(self, all_data, all_output, all_target, save_dir):
    # Create a dataloader from the concatenated data
    concatenated_dataset = torch.utils.data.TensorDataset(all_data, all_target)
    concatenated_loader = torch.utils.data.DataLoader(concatenated_dataset, batch_size=1, shuffle=False)

    # Generate and save individual heatmaps
    heatmap_paths = []
    for i, (images, targets) in enumerate(concatenated_loader):
        for j in range(images.shape[0]):
            image = images[j]
            target = targets[j]
            image_name = f'image_{i * len(images) + j}'
            heatmap_path = self.generate_individual_heatmap(image, target, save_dir, image_name)
            heatmap_paths.append(heatmap_path)

    # Combine individual heatmaps into one
    combined_heatmap_path = self.combine_heatmaps(heatmap_paths, save_dir)
    return combined_heatmap_path

if __name__ == '__main__':
    # Example usage
    # Assuming `model` is your ResNet50 model and `test_loader` is your DataLoader for testing images
    # Set up the save directory
    save_dir = 'results'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Generate and combine heatmaps
    generate_and_combine_heatmaps(model, test_loader, save_dir)

