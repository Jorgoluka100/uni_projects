"""Minimal Grad-CAM implementation for inspecting image-classifier decisions."""
from __future__ import annotations


def gradcam_heatmap(model, image, target_layer, class_index: int | None = None):
    """Return a normalized Grad-CAM heatmap for one image tensor.

    Parameters
    ----------
    model:
        PyTorch classifier in evaluation mode.
    image:
        Tensor shaped ``[1, C, H, W]``.
    target_layer:
        Convolutional module whose activations should be explained.
    class_index:
        Optional target class. Defaults to the model's predicted class.
    """
    import torch
    import torch.nn.functional as F

    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape [1, C, H, W]")

    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(image)
        target = int(logits.argmax(dim=1).item()) if class_index is None else int(class_index)
        if target < 0 or target >= logits.shape[1]:
            raise ValueError("class_index is outside the classifier output range")
        logits[0, target].backward()

        feature_map = activations[-1]
        gradient = gradients[-1]
        weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * feature_map).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam -= cam.min()
        denominator = cam.max().clamp_min(1e-12)
        return (cam / denominator).detach().cpu()
    finally:
        forward_handle.remove()
        backward_handle.remove()
