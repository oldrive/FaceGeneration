import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def tensor2im(var):
	var = var.numpy().transpose(1, 2, 0)  # [c, h, w] ==> [h, w, c]
	# var = var.cpu().detach().numpy().transpose(1, 2, 0)  # [c, h, w] ==> [h, w, c]
	var = ((var + 1) / 2)  # [-1.0, 1.0] ==> [0.0, 1.0]
	var[var < 0] = 0
	var[var > 1] = 1
	var = var * 255
	return Image.fromarray(var.astype('uint8'))


def plot_image(image, title):
	plt.axis('off')
	plt.title(title)
	plt.imshow(image)
	plt.show()


