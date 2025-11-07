import numpy as np
from scipy.interpolate import griddata
import scipy.signal

def amp_to_loudness(amp):
    if amp <= 0:
        return -np.inf
    dB = 20 * np.log10(amp)
    loudness = (2**(dB / 10)) / 16
    return loudness

def dissonance(f1, f2, l1, l2):
    x = 0.24
    s1 = 0.0207
    s2 = 18.96
    fmin = min(f1, f2)
    fmax = max(f1, f2)
    s = x / (s1 * fmin + s2)
    p = s * (fmax - fmin)

    b1 = 3.51
    b2 = 5.75

    l12 = min(l1, l2)

    return l12 * (np.exp(-b1 * p) - np.exp(-b2 * p))

def get_dissonance_data_3d_raw(spectrum, ref_freq, max_interval, step_size_3d):
    freq_array = spectrum['freq']
    amp_array = spectrum['amp']

    num_partials = len(freq_array)
    loudness_array = [amp_to_loudness(amp) for amp in amp_array]

    r_points = []
    s_points = []
    dissonance_scores = []

    r_values_iter = np.arange(1, max_interval + step_size_3d, step_size_3d)
    s_values_iter = np.arange(1, max_interval + step_size_3d, step_size_3d)

    for r in r_values_iter:
        for s in s_values_iter:
            dissonance_score = 0

            for i in range(num_partials):
                for j in range(num_partials):
                    f1 = ref_freq * freq_array[i]
                    f2 = ref_freq * freq_array[j]
                    l1 = loudness_array[i]
                    l2 = loudness_array[j]

                    d = dissonance(f1, f2, l1, l2) + \
                        dissonance(r * f1, r * f2, l1, l2) + \
                        dissonance(f1, r * f2, l1, l2) + \
                        dissonance(s * f1, s * f2, l1, l2) + \
                        dissonance(f1, s * f2, l1, l2) + \
                        dissonance(r * f1, s * f2, l1, l2)

                    dissonance_score += d

            dissonance_score /= 2
            r_points.append(r)
            s_points.append(s)
            dissonance_scores.append(dissonance_score)

    return np.array(r_points), np.array(s_points), np.array(dissonance_scores)

def transform_and_interpolate_to_triangle(r_raw, s_raw, z_raw, max_interval, grid_resolution, z_axis_ramp, std_dev_cents):
    c1 = 1200 * np.log2(r_raw)
    c2 = 1200 * np.log2(s_raw / r_raw)

    max_cents_overall = 1200 * np.log2(max_interval)
    scale_factor = 1200.0 / max_cents_overall if max_cents_overall > 0 else 1.0

    x_tri = (c1 + (c2 / 2)) * scale_factor
    y_tri = (c2 * np.sqrt(3) / 2) * scale_factor

    if len(z_raw) > 0:
        max_z_raw = np.max(z_raw)
        if max_z_raw > 0:
            z_normalized = z_raw / max_z_raw
        else:
            z_normalized = z_raw
    else:
        return np.array([]), np.array([]), np.array([])

    # Invert the normalized Z-axis values
    z_normalized = 1 - z_normalized

    # Apply the Z-axis ramp transformation
    z_transformed = np.power(np.maximum(-1, z_normalized), z_axis_ramp)

    # Use fixed range for interpolation grid based on target 0-1200 x-span
    xi = np.linspace(0, 1200, grid_resolution)
    yi = np.linspace(0, 1200 * np.sqrt(3) / 2, grid_resolution)
    XI, YI = np.meshgrid(xi, yi)

    points = np.vstack((x_tri, y_tri)).T
    values = z_transformed # Use transformed values for interpolation

    interpolated_z = griddata(points, values, (XI, YI), method='linear')

    # Apply Gaussian smoothing
    if std_dev_cents > 0:
        # Convert std_dev_cents to pixels for the current grid resolution
        # Approximate cents per pixel based on the x-axis range (now fixed to 1200)
        cents_range_x = 1200
        pixels_per_cent_x = grid_resolution / cents_range_x
        std_dev_pixels = std_dev_cents * pixels_per_cent_x

        # Create a 2D Gaussian kernel
        s_range = round(std_dev_pixels * 5) # 5 standard deviations for kernel size
        x_s = np.arange(-s_range, s_range + 1, 1)
        y_s = np.arange(-s_range, s_range + 1, 1)
        xv_s, yv_s = np.meshgrid(x_s, y_s)
        gaussian_kernel = np.exp(-((xv_s**2 + yv_s**2) / (2 * std_dev_pixels**2)))
        gaussian_kernel /= gaussian_kernel.sum() # Normalize kernel

        # Convolve with the interpolated data
        # Fill NaNs for convolution, then re-mask
        smoothed_z = interpolated_z.copy()
        nan_mask = np.isnan(smoothed_z)
        smoothed_z[nan_mask] = 0 # Temporarily fill NaNs with 0

        smoothed_z = scipy.signal.convolve2d(smoothed_z, gaussian_kernel, mode='same', boundary='symm')
        interpolated_z = smoothed_z # Update interpolated_z with smoothed data
        interpolated_z[nan_mask] = np.nan # Restore NaNs where they were originally


    # The mask now uses the fixed 1200 x-span
    mask_triangle = (YI >= -1e-9) & \
                    (YI <= (XI * np.sqrt(3)) + 1e-9) & \
                    (YI <= ((1200 - XI) * np.sqrt(3)) + 1e-9)

    interpolated_z[~mask_triangle] = np.nan

    return XI, YI, interpolated_z
