import math
import numpy as np
import scipy.signal
from PyQt5.QtGui import QImage
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import io

def generate_triangle_image(equave_ratio, width, height):
    """
    Generates a triangle image based on harmonic entropy calculations.
    """
    n_limit = 300
    c_limit = 27_000_000
    
    f = []
    equave_ratio_float = float(equave_ratio)
    for i in range(1, n_limit):
        j_min = i
        # Use equave ratio to determine upper bounds instead of fixed 2*i
        j_max = min(math.floor(i * equave_ratio_float), c_limit // i)
        if j_max < j_min:
            continue
        for j in range(j_min, j_max + 1):
            k_min = j
            k_max = min(math.floor(i * equave_ratio_float), c_limit // (i * j))
            if k_max < k_min:
                continue
            for k in range(k_min, k_max + 1):
                if i * j * k < c_limit and math.gcd(math.gcd(i, j), k) == 1:
                    f.append([i, j, k])
    f = np.array(f)

    def cents(x):
        return 1200 * np.log2(x)

    w = 1.0 / np.sqrt(np.prod(f, axis=1))

    c1 = cents(f[:, 1] / f[:, 0])
    c2 = cents(f[:, 2] / f[:, 1])

    cx = c1 + (c2 / 2)
    cy = c2 * math.sqrt(3) / 2

    # Scale to image dimensions
    max_cents = 1200 * np.log2(equave_ratio_float)
    cx = (cx / max_cents) * (width - 1)
    cy = (cy / (max_cents * math.sqrt(3) / 2)) * (height - 1)
    
    cx = np.round(cx).astype(np.int64)
    cy = np.round(cy).astype(np.int64)

    # Filter out-of-bounds coordinates
    mask = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
    cx, cy, w = cx[mask], cy[mask], w[mask]

    c = (cy, cx)

    alpha = 7
    
    k = np.zeros(shape=(height, width), dtype=np.float64)
    k_a = np.zeros(shape=(height, width), dtype=np.float64)

    np.add.at(k, c, w)
    np.add.at(k_a, c, w**alpha)

    std = 15
    s_range = round(std * 5)
    x_s = np.arange(-s_range, s_range, 1)
    y_s = np.arange(-s_range, s_range, 1)
    xv, yv = np.meshgrid(x_s, y_s)
    s = np.exp(-((xv**2 + yv**2) / (2 * std**2)))

    prod_k_s = scipy.signal.convolve(k, s, mode='same')
    prod_k_s_alpha = scipy.signal.convolve(k_a, s**alpha, mode='same')

    eps = 1e-16
    entropy = (1 / (1 - alpha)) * np.log((eps + prod_k_s_alpha) / (eps + prod_k_s**alpha))

    entropy2 = 7 - entropy
    
    # Masking the triangle shape
    x_coords = np.arange(0, width, 1)
    y_coords = np.arange(0, height, 1)
    xv_mask, yv_mask = np.meshgrid(x_coords, y_coords)
    
    # This mask is for a triangle with one horizontal side at the bottom.
    # The vertices are (0,0), (width, 0), (width/2, height)
    # The line from (0,0) to (width/2, height) is y = (2*height/width) * x
    # The line from (width,0) to (width/2, height) is y = (-2*height/width) * x + 2*height
    
    mask = yv_mask >= 0 # bottom edge
    mask &= yv_mask <= (2 * height / width) * xv_mask # Left edge
    mask &= yv_mask <= (-2 * height / width) * xv_mask + 2*height # Right edge
    
    entropy2[~mask] = np.nan # Use NaN for transparent areas

    # Blue theme custom colormap gradient
    colors = ["#23262F", "#1E1861", "#1A0EBE", "#0437f2", "#7895fc", "#A7C6ED", "#D0E1F9", 
              "#F0F4FF", "#FFFFFF"]
    custom_cmap = LinearSegmentedColormap.from_list("color_gradient", colors)

    # Use Matplotlib to generate the image with custom colormap
    fig, ax = plt.subplots(figsize=(width/200, height/200), dpi=600)
    ax.imshow(entropy2, cmap=custom_cmap, interpolation='nearest', origin='lower')
    ax.axis('off')
    fig.tight_layout(pad=0)

    # Render the figure to a buffer
    buf = io.BytesIO()
    fig.savefig(buf, format='png', transparent=True)
    buf.seek(0)
    q_image = QImage.fromData(buf.read())
    plt.close(fig)

    return q_image
