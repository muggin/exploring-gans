import matplotlib.pyplot as plt

from time import strftime

def plot_images(images, image_dims, max_col=4):
    if image_dims[-1] == 1:
	image_dims = image_dims[:-1]
    
    fig = plt.figure()
    rows = (len(images)-1)// max_col + 1
    for ix, image in enumerate(images):
        plt.subplot(rows, max_col, ix+1)
        plt.imshow(image.reshape(image_dims), cmap='gray_r', interpolation='none')
	plt.xticks([])
	plt.yticks([])
    plt.tight_layout()
    return fig

def plot_performance(*metrics):
    """
    function plots various performance metrics over time.
    provide multiple tuples in the format (metric-name, metric).
    different metrics (tuples) should be provided as consecutive arguments
    """
    fig = plt.figure()
    for label, metric in metrics:
        plt.plot(metric, label=label)
    plt.grid()
    plt.title('GAN training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_name = 'performance_{}.png'.format(time_now())
    return fig

def show_images(images, image_dims, max_col=4):
    if image_dims[-1] == 1:
	image_dims = image_dims[:-1]
    
    rows = (len(images)-1)// max_col + 1
    for ix, image in enumerate(images):
        plt.subplot(rows, max_col, ix+1)
        plt.imshow(image.reshape(image_dims), cmap='gray_r', interpolation='none')
	plt.xticks([])
	plt.yticks([])
    plt.tight_layout()
    plt.show()

def show_performance(*metrics):
    """
    function plots various performance metrics over time.
    provide multiple tuples in the format (metric-name, metric).
    different metrics (tuples) should be provided as consecutive arguments
    """
    fig = plt.figure()
    for label, metric in metrics:
        plt.plot(metric, label=label)
    plt.grid()
    plt.title('GAN training metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig_name = 'performance_{}.png'.format(time_now())
    return fig

def time_now():
    return str(strftime("%Y-%m-%d  %H:%M:%S"))
