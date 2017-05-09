import matplotlib.pyplot as plt

def plot_images(images, image_dims, max_col=4, file_name='', to_file=False):
    rows = (len(images)-1)// max_col + 1
    for ix, image in enumerate(images):
        plt.subplot(rows, max_col, ix+1)
        plt.imshow(image.reshape(image_dims), cmap='gray_r', interpolation=None)
	plt.xticks([])
	plt.yticks([])
    plt.tight_layout()
    plt.show()

def plot_performance(*metrics):
    """
    function plots various performance metrics over time.
    provide multiple tuples in the format (metric-name, metric).
    different metrics (tuples) should be provided as consecutive arguments
    """
    plt.figure()
    for label, metric in metrics:
        plt.plot(metric, label=label)
    plt.grid()
    plt.title('GAN training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
