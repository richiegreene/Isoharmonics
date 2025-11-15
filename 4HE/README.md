# 4-Note Harmonic Entropy Visualizer (4HE)

A standalone Python application for visualizing 4-note chords (tetrads) in a 3D tetrahedral space based on harmonic entropy principles. This tool allows users to explore the relationships between just intonation ratios through various configurable views and complexity measures.

## Features

The visualizer provides a control panel with the following options:

- **View Modes**:
  - **Scatter Plot**: Displays chords as points in the 3D space. The size of the points can be scaled based on their complexity.
  - **Labels**: Displays the integer ratios of the chords directly in the 3D space. The font size can be scaled based on complexity.
  - **Volume Data**: Renders a 3D volumetric representation of the harmonic entropy data.

- **Data Filtering & Generation**:
  - **Limit Mode**: Generate chords based on either an "Odd Limit" or a simple "Integer Limit".
  - **Odd-Limit / Integer-Limit**: Set the numerical limit for chord generation.
  - **Omit Unisons**: When checked, removes chords containing duplicate integers (e.g., `4:5:5:6`).
  - **Omit Octaves**: When checked, removes chords containing octave-multiples (e.g., `3:4:5:6`).

- **Complexity & Scaling**:
  - **Complexity Measures**: Choose a formula to define the "complexity" of each chord, which in turn controls the size of the visual elements.
    - *Gradus*
    - *Tenney*
    - *Weil*
    - *Wilson*
    - *Off* (disables complexity-based scaling)
  - **Size**: A global multiplier to adjust the overall size of all points or labels.
  - **Feature Scaling**: Controls the proportional difference between the smallest and largest elements. A higher value creates a more dramatic size difference.

## Requirements

This application is built with Python and requires the following packages:

- `PyQt5`
- `numpy`
- `scipy`
- `pyqtgraph`

A `requirements.txt` file is included for easy installation.

## Installation

1.  **Clone or download the `4HE` directory.**

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the application, navigate to the `4HE` directory in your terminal and run:

```bash
python main.py
```
