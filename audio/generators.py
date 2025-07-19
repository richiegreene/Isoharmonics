import numpy as np

bits = 16
sample_rate = 44100
fade_duration = 0.5
max_amplitude = 2 ** (bits - 1) - 1
normalization_factor = 0.125
all_playing_sounds = []

def get_dynamic_normalization(roll_off_rate):
    """Safe dynamic normalization calculation for both roll-off directions"""
    try:
        base_factor = 0.125
        if roll_off_rate < -0.0001:
            return max(0.001, base_factor * (10 ** (-1 * abs(roll_off_rate) / 3)))
        return base_factor
    except:
        return base_factor

def generate_combined_playback_buffer(frequencies, ratios, duration, roll_off_rate, phase_factor=0.0):
    """Generate playback buffer with proper negative roll-off handling"""
    try:
        num_samples = int(duration * sample_rate)
        fade_samples = int(fade_duration * sample_rate)
        t = np.linspace(0, duration, num_samples, False)
        combined = np.zeros(num_samples, dtype=np.float32)
        
        for freq, ratio in zip(frequencies, ratios):
            effective_ratio = max(0.0001, float(ratio))
            
            if abs(roll_off_rate) > 0.0001:
                if roll_off_rate < 0:
                    amp = effective_ratio ** abs(roll_off_rate)
                else:
                    amp = 1.0 / (effective_ratio ** roll_off_rate)
            else:
                amp = 1.0
                
            combined += np.sin(2 * np.pi * freq * t + phase_factor) * amp
        
        if num_samples > fade_samples * 2:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            combined[:fade_samples] *= fade_in
            combined[-fade_samples:] *= fade_out
        
        max_val = np.max(np.abs(combined))
        if max_val > 0:
            combined /= max_val
        
        scaled = (combined * normalization_factor * max_amplitude).astype(np.int16)
        
        stereo_buffer = np.zeros((num_samples, 2), dtype=np.int16)
        stereo_buffer[:, 0] = scaled
        stereo_buffer[:, 1] = scaled
        return stereo_buffer
        
    except Exception as e:
        print(f"Buffer generation error: {e}")
        return np.zeros((int(duration * sample_rate), 2), dtype=np.int16)

def generate_tapered_sine_wave(freq, duration, fade_duration, amplitude_factor=1.0, phase=0.0):
    num_samples = int(round(duration * sample_rate))
    fade_samples = int(round(fade_duration * sample_rate))
    t = np.linspace(0, duration, num_samples, False)
    raw_sine = np.sin(2 * np.pi * freq * t + phase)
    scaled_sine = raw_sine * (max_amplitude * normalization_factor * amplitude_factor)
    
    sine_wave = scaled_sine.astype(np.int16)
    
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)
    sine_wave[:fade_samples] = np.multiply(sine_wave[:fade_samples], fade_in)
    sine_wave[-fade_samples:] = np.multiply(sine_wave[-fade_samples:], fade_out)
    
    stereo_buffer = np.zeros((num_samples, 2), dtype=np.int16)
    stereo_buffer[:, 0] = sine_wave
    stereo_buffer[:, 1] = sine_wave
    return stereo_buffer
