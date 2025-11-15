import * as THREE from 'https://unpkg.com/three@0.126.0/build/three.module.js';
import { OrbitControls } from 'https://unpkg.com/three@0.126.0/examples/jsm/controls/OrbitControls.js';

let scene, camera, renderer, controls;
let pyodide;
let loadingOverlay = document.getElementById('loading-overlay');
let python_ready = false;

async function initPyodide() {
    loadingOverlay.style.display = 'flex';
    pyodide = await loadPyodide({
        indexURL: "https://cdn.jsdelivr.net/pyodide/v0.25.0/full/",
    });
    console.log("Pyodide loaded.");

    // Load necessary Python packages
    await pyodide.loadPackage(["numpy", "scipy"]);
    console.log("Numpy and Scipy loaded.");

    // Create directories in Pyodide's virtual filesystem
    pyodide.FS.mkdir("python");
    pyodide.FS.mkdir("python/theory");

    // Write the Python module contents to Pyodide's virtual filesystem
    const tetrahedron_generator_py_content = `import math
import numpy as np
import scipy.signal
from itertools import combinations_with_replacement
from fractions import Fraction
from theory.calculations import get_odd_limit, _generate_valid_numbers, calculate_complexity

def cents(x):
    return 1200 * np.log2(x)

def generate_tetrahedron_data(equave_ratio, resolution):
    n_limit = 60
    c_limit = 2_000_000

    f = []
    equave_ratio_float = float(equave_ratio)
    
    for i in range(1, n_limit):
        j_min = i
        j_max = min(math.floor(i * equave_ratio_float) + 1, int(pow(c_limit / i, 1/3)) + 1)
        if j_max < j_min: continue
        for j in range(j_min, j_max):
            k_min = j
            k_max = min(math.floor(j * equave_ratio_float) + 1, int(pow(c_limit / (i * j), 1/2)) + 1)
            if k_max < k_min: continue
            for k in range(k_min, k_max):
                l_min = k
                l_max = min(math.floor(k * equave_ratio_float) + 1, c_limit // (i * j * k) + 1)
                if l_max < l_min: continue
                for l in range(l_min, l_max):
                    if math.gcd(math.gcd(math.gcd(i, j), k), l) == 1:
                        f.append([i, j, k, l])
    
    if not f:
        return None, None, None, None

    f = np.array(f, dtype=np.float64)

    w = 1.0 / np.sqrt(np.prod(f, axis=1))

    c1 = cents(f[:, 1] / f[:, 0])
    c2 = cents(f[:, 2] / f[:, 1])
    c3 = cents(f[:, 3] / f[:, 2])

    max_cents = 1200 * np.log2(equave_ratio_float)
    
    cx = np.round((c1 / max_cents) * (resolution - 1)).astype(int)
    cy = np.round((c2 / max_cents) * (resolution - 1)).astype(int)
    cz = np.round((c3 / max_cents) * (resolution - 1)).astype(int)

    mask = (cx >= 0) & (cx < resolution) & (cy >= 0) & (cy < resolution) & (cz >= 0) & (cz < resolution)
    cx, cy, cz, w = cx[mask], cy[mask], cz[mask], w[mask]

    coords = (cz, cy, cx)

    alpha = 7
    
    k = np.zeros(shape=(resolution, resolution, resolution), dtype=np.float64)
    k_a = np.zeros(shape=(resolution, resolution, resolution), dtype=np.float64)

    np.add.at(k, coords, w)
    np.add.at(k_a, coords, w**alpha)

    std = 2.0
    s_range = round(std * 2)
    x_s, y_s, z_s = np.mgrid[-s_range:s_range+1, -s_range:s_range+1, -s_range:s_range+1]
    s_kernel = np.exp(-((x_s**2 + y_s**2 + z_s**2) / (2 * std**2)))

    prod_k_s = scipy.signal.convolve(k, s_kernel, mode='same')
    prod_k_s_alpha = scipy.signal.convolve(k_a, s_kernel**alpha, mode='same')

    eps = 1e-16
    entropy = (1 / (1 - alpha)) * np.log((eps + prod_k_s_alpha) / (eps + prod_k_s**alpha))
    
    entropy[np.isnan(entropy)] = 0
    entropy = np.nanmax(entropy) - entropy

    c1_grid, c2_grid, c3_grid = np.mgrid[0:max_cents:complex(0, resolution), 0:max_cents:complex(0, resolution), 0:max_cents:complex(0, resolution)]
    
    mask = c1_grid + c2_grid + c3_grid > max_cents
    mask = np.transpose(mask, (2, 1, 0))

    mask |= (k == 0)
    
    entropy[mask] = np.nan

    return c1_grid, c2_grid, c3_grid, entropy

def generate_odd_limit_points(limit_value, equave_ratio, limit_mode="odd", complexity_measure="Tenney", hide_unison_voices=False, omit_octaves=False):
    points = []
    equave_ratio_float = float(equave_ratio)
    
    valid_numbers = _generate_valid_numbers(limit_value, limit_mode)

    if not valid_numbers:
        return []

    sorted_valid_numbers = sorted(list(valid_numbers))
    
    for combo in combinations_with_replacement(sorted_valid_numbers, 4):
        if hide_unison_voices and len(set(combo)) < 4:
            continue

        if omit_octaves:
            has_octave = False
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if combo[j] == combo[i] * 2:
                        has_octave = True
                        break
                if has_octave:
                    break
            if has_octave:
                continue

        i, j, k, l = combo
        
        if l / i > equave_ratio_float:
            continue
            
        if math.gcd(math.gcd(math.gcd(i, j), k), l) != 1:
            continue
            
        if limit_mode == "odd":
            if (get_odd_limit(Fraction(j, i)) > limit_value or
                get_odd_limit(Fraction(k, j)) > limit_value or
                get_odd_limit(Fraction(l, k)) > limit_value):
                continue
        
        c1 = cents(j / i)
        c2 = cents(k / j)
        c3 = cents(l / k)
        
        complexity = max(
            calculate_complexity(complexity_measure, Fraction(j, i)),
            calculate_complexity(complexity_measure, Fraction(k, j)),
            calculate_complexity(complexity_measure, Fraction(l, k))
        )
        
        points.append((c1, c2, c3, complexity))
        
    return points
`;
    pyodide.FS.writeFile("python/tetrahedron_generator.py", tetrahedron_generator_py_content, { encoding: "utf8" });

    const calculations_py_content = `import math
from fractions import Fraction
from functools import reduce
from math import gcd
from itertools import combinations_with_replacement
import numpy as np

def cents(x):
    return 1200 * np.log2(x)

def calculate_edo_step(cents, edo):
    step_size = 1200 / edo
    step = round(cents / step_size)
    error = step * step_size - cents
    step_str = f"-{abs(step)}" if step < 0 else str(step)
    return step_str, error

def calculate_12edo_step(cents):
    step_size = 1200 / 12
    step = round(cents / step_size)
    error = step * step_size - cents
    return step, error

def ratio_to_cents(ratio):
    return 1200 * math.log2(ratio)

def generate_iso_series(fundamental, isoharmonic, partials_above, partials_below):
    series = []
    current_ratio = isoharmonic
    for _ in range(partials_below):
        current_ratio = current_ratio - fundamental
        series.insert(0, current_ratio)
    series.append(isoharmonic)
    current_ratio = isoharmonic
    for _ in range(partials_above):
        current_ratio = current_ratio + fundamental
        series.append(current_ratio)
    return series

def find_gcd(list):
    return reduce(gcd, list)

def find_lcd(denominators):
    def lcm(a, b):
        return a * b // gcd(a, b)
    return reduce(lcm, denominators)

def format_series_segment(series):
    fractions = [Fraction(ratio).limit_denominator() for ratio in series]
    denominators = [frac.denominator for frac in fractions]
    lcd = find_lcd(denominators)
    numerators = [int(frac.numerator * (lcd / frac.denominator)) for frac in fractions] # Fixed: use frac.denominator instead of hardcoded 1
    
    if lcd == 1:
        return ':'.join(map(str, numerators))
    else:
        return f"({':'.join(map(str, numerators))})/{lcd}"

def simplify_ratio(ratio):
    frac = Fraction(ratio).limit_denominator()
    return f"{frac.numerator}/{frac.denominator}"

def get_odd_part_of_number(num):
    if num == 0:
        return 0
    while num > 0 and num % 2 == 0:
        num //= 2
    return num

def get_odd_limit(ratio):
    try:
        ratio = Fraction(ratio).limit_denominator(10000)
        n, d = ratio.numerator, ratio.denominator
        
        n_odd_part = get_odd_part_of_number(n)
        d_odd_part = get_odd_part_of_number(d)
            
        return max(n_odd_part, d_odd_part)
    except (ValueError, ZeroDivisionError):
        return 1

def get_integer_limit(ratio):
    try:
        ratio = Fraction(ratio).limit_denominator(10000)
        return max(ratio.numerator, ratio.denominator)
    except (ValueError, ZeroDivisionError):
        return 1

def get_prime_factorization(n):
    factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
       factors.append(n)
    return factors

def tenney_norm(ratio):
    ratio = Fraction(ratio).limit_denominator(10000)
    return math.log2(ratio.numerator * ratio.denominator)

def weil_norm(ratio):
    ratio = Fraction(ratio).limit_denominator(10000)
    return math.log2(max(ratio.numerator, ratio.denominator))

def wilson_norm(ratio):
    ratio = Fraction(ratio).limit_denominator(10000)
    factors_n = get_prime_factorization(ratio.numerator)
    factors_d = get_prime_factorization(ratio.denominator)
    return sum(factors_n) + sum(factors_d)

def gradus_norm(ratio):
    ratio = Fraction(ratio).limit_denominator(10000)
    factors_n = get_prime_factorization(ratio.numerator)
    factors_d = get_prime_factorization(ratio.denominator)
    s = sum(factors_n) + sum(factors_d)
    n = len(factors_n) + len(factors_d)
    return s - n + 1

def calculate_complexity(complexity_measure, ratio):
    if complexity_measure == "Tenney":
        return tenney_norm(ratio)
    elif complexity_measure == "Weil":
        return weil_norm(ratio)
    elif complexity_measure == "Wilson":
        return wilson_norm(ratio)
    elif complexity_measure == "Gradus":
        return gradus_norm(ratio)
    else:
        return 0

def _generate_valid_numbers(limit_value, limit_mode):
    """
    Generates a set of valid numbers based on the limit mode.
    """
    valid_numbers = set()
    if limit_mode == "odd":
        max_num_to_check = max(limit_value * 2, 100)
        for num in range(1, max_num_to_check + 1):
            if get_odd_part_of_number(num) <= limit_value:
                valid_numbers.add(num)
    elif limit_mode == "integer":
        valid_numbers = set(range(1, limit_value + 1))
    return valid_numbers

def generate_ji_tetra_labels(limit_value, equave_ratio, limit_mode="odd", complexity_measure="Tenney", hide_unison_voices=False, omit_octaves=False):
    labels_data = []
    equave_ratio_float = float(equave_ratio)

    valid_numbers = _generate_valid_numbers(limit_value, limit_mode)
            
    if not valid_numbers:
        return []

    sorted_valid_numbers = sorted(list(valid_numbers))
    
    for combo in combinations_with_replacement(sorted_valid_numbers, 4):
        if hide_unison_voices and len(set(combo)) < 4:
            continue

        if omit_octaves:
            has_octave = False
            for i in range(len(combo)):
                for j in range(i + 1, len(combo)):
                    if combo[j] == combo[i] * 2:
                        has_octave = True
                        break
                if has_octave:
                    break
            if has_octave:
                continue

        i, j, k, l = combo
        
        if l / i > equave_ratio_float:
            continue
            
        if gcd(gcd(gcd(i, j), k), l) != 1:
            continue

        if limit_mode == "odd":
            if (get_odd_limit(Fraction(j, i)) > limit_value or
                get_odd_limit(Fraction(k, j)) > limit_value or
                get_odd_limit(Fraction(l, k)) > limit_value):
                continue
            
        c1 = cents(j / i)
        c2 = cents(k / j)
        c3 = cents(l / k)
        
        complexity = max(
            calculate_complexity(complexity_measure, Fraction(j, i)),
            calculate_complexity(complexity_measure, Fraction(k, j)),
            calculate_complexity(complexity_measure, Fraction(l, k))
        )
        
        label = f"{i}:{j}:{k}:{l}"
        
        labels_data.append(((c1, c2, c3), label, complexity))
        
    return labels_data

def get_primes_less_than_or_equal_to(p):
    primes = []
    for num in range(2, p + 1):
        is_prime = True
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

def get_max_exponent_for_p_smooth(n, p_limit, primes=None):
    if primes is None:
        primes = get_primes_less_than_or_equal_to(p_limit)
    
    max_exp = 0
    
    temp_n = n
    for p in primes:
        if temp_n == 1:
            break
        if temp_n % p == 0:
            exp = 0
            while temp_n % p == 0:
                exp += 1
                temp_n //= p
            max_exp = max(max_exp, exp)
            
    if temp_n > 1:
        return float('inf')
        
    return max_exp

def generate_ji_triads(limit_value, equave=Fraction(2,1), limit_mode="odd", prime_limit=7, max_exponent=4):
    if limit_value < 1 and limit_mode != "prime":
        return []

    valid_intervals = set([Fraction(1,1)])
    
    max_val_for_n_d = 0
    if limit_mode == "odd" or limit_mode == "integer":
        max_val_for_n_d = limit_value * 3
    elif limit_mode == "prime":
        max_val_for_n_d = prime_limit * max_exponent * 3

    primes = None
    if limit_mode == "prime":
        primes = get_primes_less_than_or_equal_to(prime_limit)

    for n_val in range(1, max_val_for_n_d + 1):
        for d_val in range(1, max_val_for_n_d + 1):
            if n_val == 0 or d_val == 0: continue
            ratio = Fraction(n_val, d_val)
            
            if limit_mode == "odd":
                if get_odd_limit(ratio) <= limit_value:
                    valid_intervals.add(ratio)
            elif limit_mode == "integer":
                if get_integer_limit(ratio) <= limit_value:
                    valid_intervals.add(ratio)
            elif limit_mode == "prime":
                num_exp = get_max_exponent_for_p_smooth(ratio.numerator, prime_limit, primes)
                den_exp = get_max_exponent_for_p_smooth(ratio.denominator, prime_limit, primes)
                if num_exp <= max_exponent and den_exp <= max_exponent:
                    valid_intervals.add(ratio)

    if limit_mode == "odd":
        if get_odd_limit(equave) <= limit_value:
            valid_intervals.add(equave)
    elif limit_mode == "integer":
        if get_integer_limit(equave) <= limit_value:
            valid_intervals.add(equave)
    elif limit_mode == "prime":
        num_exp = get_max_exponent_for_p_smooth(equave.numerator, prime_limit, primes)
        den_exp = get_max_exponent_for_p_smooth(equave.denominator, prime_limit, primes)
        if num_exp <= max_exponent and den_exp <= max_exponent:
            valid_intervals.add(equave)

    sorted_intervals = sorted(list(valid_intervals))

    triads = []
    triad_labels = set()

    for i in range(len(sorted_intervals)):
        r1 = sorted_intervals[i]
        for j in range(i, len(sorted_intervals)):
            r2 = sorted_intervals[j]
            
            r3 = r2 / r1
            
            cx_ratio = None
            cy_ratio = None

            if limit_mode == "odd":
                if get_odd_limit(r3) <= limit_value:
                    cx_ratio = r1
                    cy_ratio = r3
            elif limit_mode == "integer":
                if get_integer_limit(r3) <= limit_value:
                    cx_ratio = r1
                    cy_ratio = r3
            elif limit_mode == "prime":
                num_exp = get_max_exponent_for_p_smooth(r3.numerator, prime_limit, primes)
                den_exp = get_max_exponent_for_p_smooth(r3.denominator, prime_limit, primes)
                if num_exp <= max_exponent and den_exp <= max_exponent:
                    cx_ratio = r1
                    cy_ratio = r3

            if cx_ratio is None or cy_ratio is None: continue

            if cx_ratio < 1 or cy_ratio < 1: continue

            cx = 1200 * math.log2(cx_ratio)
            cy = 1200 * math.log2(cy_ratio)

            if cx + cy > 1200 * math.log2(equave) + 1e-9: continue

            common_denom = r1.denominator * r2.denominator
            a = common_denom
            b = r1.numerator * r2.denominator
            c = r2.numerator * r1.denominator
            
            common_divisor = gcd(gcd(a,b),c)
            sa, sb, sc = a//common_divisor, b//common_divisor, c//common_divisor
            
            sorted_triad = sorted([sa, sb, sc])
            label = f"{sorted_triad[0]}:{sorted_triad[1]}:{sorted_triad[2]}"

            if label not in triad_labels:
                triads.append(((cx, cy), label))
                triad_labels.add(label)

    return triads
`;
    pyodide.FS.writeFile("python/theory/calculations.py", calculations_py_content, { encoding: "utf8" });
    pyodide.FS.writeFile("python/theory/__init__.py", "", { encoding: "utf8" }); // Create __init__.py

    // Add current directory to Python path
    pyodide.runPython("import sys; sys.path.append('./python')");
    await pyodide.loadPackage("micropip"); // Install micropip first
    console.log("Micropip loaded.");
    // The 'fractions' module is part of Python's standard library and does not need micropip.install.
    // It's included with Pyodide by default.

    python_ready = true;
    loadingOverlay.style.display = 'none';

    initThreeJS();
    animate();
    
    const default_limit_value = 5;
    const default_equave_ratio = 2; // Default to octave
    const default_complexity_method = "Tenney"; // Default complexity method
    await updateTetrahedron(default_limit_value, default_equave_ratio, default_complexity_method);

    // Add event listener for the update button
    document.getElementById('updateButton').addEventListener('click', async () => {
        const limitValue = parseFloat(document.getElementById('limitValue').value);
        const equaveRatio = parseFloat(document.getElementById('equaveRatio').value);
        const complexityMethod = document.getElementById('complexityMethod').value;
        if (!isNaN(limitValue) && !isNaN(equaveRatio)) {
            await updateTetrahedron(limitValue, equaveRatio, complexityMethod);
        } else {
            console.error("Invalid input for limit value or equave ratio.");
        }
    });
}

function initThreeJS() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x222222);

    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    camera.position.z = 5;

    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.getElementById('container').appendChild(renderer.domElement);

    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.25;
    controls.screenSpacePanning = false;
    controls.minDistance = 1;
    controls.maxDistance = 10;

    window.addEventListener('resize', onWindowResize, false);
}

function onWindowResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
}

function animate() {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
}

function makeTextSprite(message, parameters) {
    if ( parameters === undefined ) parameters = {};
    const fontface = parameters.hasOwnProperty("fontface") ? parameters["fontface"] : "Arial";
    const fontsize = parameters.hasOwnProperty("fontsize") ? parameters["fontsize"] : 50; // Increased for better resolution
    const borderThickness = parameters.hasOwnProperty("borderThickness") ? parameters["borderThickness"] : 4;
    const borderColor = parameters.hasOwnProperty("borderColor") ? parameters["borderColor"] : { r:0, g:0, b:0, a:1.0 };
    const backgroundColor = parameters.hasOwnProperty("backgroundColor") ? parameters["backgroundColor"] : { r:255, g:255, b:255, a:0.0 };
    const textColor = parameters.hasOwnProperty("textColor") ? parameters["textColor"] : { r:255, g:255, b:255, a:1.0 };

    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    context.font = "Bold " + fontsize + "px " + fontface;
    
    // get size data (height depends only on font size)
    const metrics = context.measureText( message );
    const textWidth = metrics.width;

    // background color
    context.fillStyle   = "rgba(" + backgroundColor.r + "," + backgroundColor.g + "," + backgroundColor.b + "," + backgroundColor.a + ")";
    // border color
    context.strokeStyle = "rgba(" + borderColor.r + "," + borderColor.g + "," + borderColor.b + "," + borderColor.a + ")";

    context.lineWidth = borderThickness;
    roundRect(context, borderThickness/2, borderThickness/2, textWidth + borderThickness, fontsize * 1.4 + borderThickness, 6);
    // 1.4 is font height factor

    // text color
    context.fillStyle = "rgba(" + textColor.r + ", " + textColor.g + ", " + textColor.b + ", 1.0)";
    context.fillText( message, borderThickness, fontsize + borderThickness);

    // canvas contents will be used for a texture
    const texture = new THREE.Texture(canvas) 
    texture.needsUpdate = true;

    const spriteMaterial = new THREE.SpriteMaterial( { map: texture } );
    const sprite = new THREE.Sprite( spriteMaterial );
    sprite.scale.set(1.0 * textWidth/fontsize, 1.4 * fontsize/fontsize, 1.0); // Scale sprite to fit text
    return sprite;  
}

// function for drawing rounded rectangles
function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x+r, y);
    ctx.lineTo(x+w-r, y);
    ctx.quadraticCurveTo(x+w, y, x+w, y+r);
    ctx.lineTo(x+w, y+h-r);
    ctx.quadraticCurveTo(x+w, y+h, x+w-r, y+h);
    ctx.lineTo(x+r, y+h);
    ctx.quadraticCurveTo(x, y+h, x, y+h-r);
    ctx.lineTo(x, y+r);
    ctx.quadraticCurveTo(x, y, x+r, y);
    ctx.closePath();
    ctx.fill();
    ctx.stroke();   
}

async function updateTetrahedron(limit_value, equave_ratio, complexity_method) {
    if (!python_ready) {
        console.warn("Python environment not ready yet.");
        return;
    }
    
    // Clear previous points and labels
    while(scene.children.length > 0){ 
        scene.remove(scene.children[0]); 
    }

    // Call the Python function to get points (c1, c2, c3, complexity)
    // and labels (c1, c2, c3), label, complexity
    const points_py_code = `
        from tetrahedron_generator import generate_odd_limit_points
        generate_odd_limit_points(
            limit_value=${limit_value}, 
            equave_ratio=${equave_ratio}, 
            limit_mode="odd", 
            complexity_measure="${complexity_method}", 
            hide_unison_voices=False, 
            omit_octaves=False
        )
    `;
    const labels_py_code = `
        from theory.calculations import generate_ji_tetra_labels
        generate_ji_tetra_labels(
            limit_value=${limit_value}, 
            equave_ratio=${equave_ratio}, 
            limit_mode="odd", 
            complexity_measure="${complexity_method}", 
            hide_unison_voices=False, 
            omit_octaves=False
        )
    `;

    const raw_points_data = await pyodide.runPythonAsync(points_py_code);
    const raw_labels_data = await pyodide.runPythonAsync(labels_py_code);

    // Process raw_points_data for Three.js points
    const positions = [];
    const colors = [];
    const color = new THREE.Color();

    let minComplexity = Infinity;
    let maxComplexity = -Infinity;
    if (raw_points_data.length > 0) {
        raw_points_data.forEach(p => {
            minComplexity = Math.min(minComplexity, p[3]);
            maxComplexity = Math.max(maxComplexity, p[3]);
        });
    }

    // Create a Map for quick lookup of labels by coordinates
    const labels_map = new Map();
    raw_labels_data.forEach(label_item => {
        const coords_key = `${label_item[0][0].toFixed(2)},${label_item[0][1].toFixed(2)},${label_item[0][2].toFixed(2)}`;
        labels_map.set(coords_key, label_item[1]); // Store label string
    });


    raw_points_data.forEach(p => {
        // Normalize coordinates and complexity for visualization
        const scale = 1 / 600; // 1200 cents max, so divide by 600 to get a range of 2.

        const x = p[0] * scale;
        const y = p[1] * scale;
        const z = p[2] * scale;

        positions.push(x, y, z);

        const normalizedComplexity = (p[3] - minComplexity) / (maxComplexity - minComplexity);
        color.setHSL( (1 - normalizedComplexity) * 0.35, 1, 0.5 ); // Green to Red HSL
        colors.push(color.r, color.g, color.b);

        // Add labels
        const coords_key = `${p[0].toFixed(2)},${p[1].toFixed(2)},${p[2].toFixed(2)}`;
        const label_text = labels_map.get(coords_key);
        if (label_text) {
            const sprite = makeTextSprite(label_text, { fontsize: 30, textColor: { r:255, g:255, b:255, a:1.0 } });
            sprite.position.set(x + 0.05, y + 0.05, z); // Offset slightly from the point
            sprite.scale.set(0.2, 0.1, 1); // Adjust scale as needed
            scene.add(sprite);
        }
    });

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

    const material = new THREE.PointsMaterial({
        size: 0.05,
        vertexColors: true,
        transparent: true,
        opacity: 0.7
    });

    const points = new THREE.Points(geometry, material);
    scene.add(points);
}

initPyodide();