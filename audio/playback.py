import pygame
from fractions import Fraction
from audio.generators import generate_tapered_sine_wave, generate_combined_playback_buffer, all_playing_sounds

def play_sine_wave(freq, duration, ratio=Fraction(1,1), roll_off_rate=0.0, phase=0.0):
    try:
        effective_ratio = max(0.0001, float(ratio))
        if abs(roll_off_rate) > 0.0001:
            amplitude_factor = 1.0 / (effective_ratio ** abs(roll_off_rate))
        else:
            amplitude_factor = 1.0
            
        pygame.mixer.init()
        sound_buffer = generate_tapered_sine_wave(
            freq, 
            duration, 
            0.5, 
            amplitude_factor, 
            phase
        )
        sound = pygame.sndarray.make_sound(sound_buffer)
        sound.play()
        all_playing_sounds.append(sound)
        return sound
    except Exception as e:
        print(f"Sine wave generation error: {e}")
        return None

def play_single_sine_wave(freq, duration, phase=0.0):
    amplitude_factor = 1.0
    pygame.mixer.init()
    sound_buffer = generate_tapered_sine_wave(freq, duration, 0.5, 
                                            amplitude_factor, phase)
    sound = pygame.sndarray.make_sound(sound_buffer)
    sound.play()
    all_playing_sounds.append(sound)
    return sound

def play_combined_sine_wave(frequencies, ratios, duration, roll_off_rate, phase=0.0):
    buffer = generate_combined_playback_buffer(frequencies, ratios, duration, 
                                             roll_off_rate, phase)
    sound = pygame.sndarray.make_sound(buffer)
    channel = pygame.mixer.find_channel()
    channel.play(sound)
    return channel

def stop_sound(sound):
    if sound:
        sound.stop()

def stop_all_sounds():
    global all_playing_sounds
    for sound in all_playing_sounds:
        stop_sound(sound)
    all_playing_sounds = []
