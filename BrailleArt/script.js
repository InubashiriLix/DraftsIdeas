// Braille patterns mapping (2x4 dots)
const BRAILLE_PATTERNS = [
    [0x01, 0x08],
    [0x02, 0x10],
    [0x04, 0x20],
    [0x40, 0x80]
];

const BRAILLE_UNICODE_BASE = 0x2800;

// DOM Elements
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const previewSection = document.getElementById('previewSection');
const imagePreview = document.getElementById('imagePreview');
const convertBtn = document.getElementById('convertBtn');
const outputSection = document.getElementById('outputSection');
const brailleOutput = document.getElementById('brailleOutput');
const copyBtn = document.getElementById('copyBtn');

const widthInput = document.getElementById('widthInput');
const widthSlider = document.getElementById('widthSlider');
const heightInput = document.getElementById('heightInput');
const heightSlider = document.getElementById('heightSlider');
const colorToggle = document.getElementById('colorToggle');
const brightnessSlider = document.getElementById('brightnessSlider');
const contrastSlider = document.getElementById('contrastSlider');

let currentImage = null;

// Sync input and slider
function syncInputs(input, slider) {
    input.addEventListener('input', () => {
        slider.value = input.value;
    });
    slider.addEventListener('input', () => {
        input.value = slider.value;
    });
}

syncInputs(widthInput, widthSlider);
syncInputs(heightInput, heightSlider);

// File upload handling
imageInput.addEventListener('change', handleImageUpload);

uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file && file.type.startsWith('image/')) {
        loadImage(file);
    }
});

function handleImageUpload(e) {
    const file = e.target.files[0];
    if (file) {
        loadImage(file);
    }
}

function loadImage(file) {
    const reader = new FileReader();
    reader.onload = (e) => {
        const img = new Image();
        img.onload = () => {
            currentImage = img;
            imagePreview.src = e.target.result;
            previewSection.style.display = 'block';
            uploadArea.style.display = 'none';
            convertBtn.disabled = false;
        };
        img.src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Convert button
convertBtn.addEventListener('click', convertToBraille);

function convertToBraille() {
    if (!currentImage) return;

    const width = parseInt(widthInput.value);
    const height = parseInt(heightInput.value);
    const useColor = colorToggle.checked;
    const brightness = parseFloat(brightnessSlider.value);
    const contrast = parseFloat(contrastSlider.value);

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Each braille character represents 2x4 pixels
    canvas.width = width * 2;
    canvas.height = height * 4;

    // Draw image to canvas
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);

    // Apply brightness and contrast
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    applyFilters(imageData, brightness, contrast);
    ctx.putImageData(imageData, 0, 0);

    // Get pixel data
    const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);

    // Convert to braille
    let result = '';
    for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
            const brailleChar = getBrailleChar(pixels, x * 2, y * 4);
            
            if (useColor) {
                const color = getAverageColor(pixels, x * 2, y * 4);
                result += `<span style="color: ${color}">${brailleChar}</span>`;
            } else {
                result += brailleChar;
            }
        }
        result += '\n';
    }

    // Display result
    if (useColor) {
        brailleOutput.innerHTML = result;
    } else {
        brailleOutput.textContent = result;
    }
    outputSection.style.display = 'block';
    
    // Scroll to output
    outputSection.scrollIntoView({ behavior: 'smooth' });
}

function getBrailleChar(imageData, startX, startY) {
    let brailleCode = BRAILLE_UNICODE_BASE;

    for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 2; dx++) {
            const x = startX + dx;
            const y = startY + dy;
            
            if (x < imageData.width && y < imageData.height) {
                const idx = (y * imageData.width + x) * 4;
                const r = imageData.data[idx];
                const g = imageData.data[idx + 1];
                const b = imageData.data[idx + 2];
                
                // Calculate brightness
                const brightness = (r * 0.299 + g * 0.587 + b * 0.114);
                
                // If pixel is dark enough, add the dot
                if (brightness < 128) {
                    brailleCode |= BRAILLE_PATTERNS[dy][dx];
                }
            }
        }
    }

    return String.fromCharCode(brailleCode);
}

function getAverageColor(imageData, startX, startY) {
    let r = 0, g = 0, b = 0, count = 0;

    for (let dy = 0; dy < 4; dy++) {
        for (let dx = 0; dx < 2; dx++) {
            const x = startX + dx;
            const y = startY + dy;
            
            if (x < imageData.width && y < imageData.height) {
                const idx = (y * imageData.width + x) * 4;
                r += imageData.data[idx];
                g += imageData.data[idx + 1];
                b += imageData.data[idx + 2];
                count++;
            }
        }
    }

    if (count === 0) return 'rgb(0,0,0)';

    r = Math.round(r / count);
    g = Math.round(g / count);
    b = Math.round(b / count);

    return `rgb(${r},${g},${b})`;
}

function applyFilters(imageData, brightness, contrast) {
    const data = imageData.data;
    const factor = (259 * (contrast * 255 + 255)) / (255 * (259 - contrast * 255));

    for (let i = 0; i < data.length; i += 4) {
        // Apply brightness
        let r = data[i] * brightness;
        let g = data[i + 1] * brightness;
        let b = data[i + 2] * brightness;

        // Apply contrast
        r = factor * (r - 128) + 128;
        g = factor * (g - 128) + 128;
        b = factor * (b - 128) + 128;

        // Clamp values
        data[i] = Math.max(0, Math.min(255, r));
        data[i + 1] = Math.max(0, Math.min(255, g));
        data[i + 2] = Math.max(0, Math.min(255, b));
    }
}

// Copy to clipboard
copyBtn.addEventListener('click', async () => {
    const text = brailleOutput.textContent;
    
    try {
        await navigator.clipboard.writeText(text);
        copyBtn.textContent = 'Copied!';
        copyBtn.classList.add('copied');
        
        setTimeout(() => {
            copyBtn.textContent = 'Copy to Clipboard';
            copyBtn.classList.remove('copied');
        }, 2000);
    } catch (err) {
        console.error('Failed to copy:', err);
        alert('Failed to copy to clipboard');
    }
});
