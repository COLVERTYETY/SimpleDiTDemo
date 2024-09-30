// Dictionary holding the image sources for each diffusion step
const diffusionImages = {};
const diffusionzt = {};

// Path to the transparent gray placeholder image
const placeholderImage = "diffusion_placeholder.png"; // Fallback image

// Get references to the HTML elements
const diffusionSlider = document.getElementById('diffusionSlider');
const sliderValueDisplay = document.getElementById('sliderValue');
const imageContainer = document.getElementById('imageCascade');
const labelSlider = document.getElementById('labelSlider');
const indexValueDisplay = document.getElementById('indexValue');


// Update the label slider value display
labelSlider.addEventListener('input', () => {
    const labelValue = parseInt(labelSlider.value);
    indexValueDisplay.textContent = labelValue;
});
// Function to load the image or placeholder if missing
function loadImage(src) {
    const img = new Image();
    img.src = src;
    img.onerror = () => { img.src = placeholderImage; }; // Fallback to transparent gray image
    img.style.imageRendering = 'pixelated';
    return img;
}

// Update the images based on the slider value
diffusionSlider.addEventListener('input', () => {
    const step = parseInt(diffusionSlider.value);
    sliderValueDisplay.textContent = step;

    // Update the image view with current and cascading previous images
    updateImageDisplay(step);
});

// Function to update the image display with cascading previous images
function updateImageDisplay(currentIndex) {
    // Clear previous images
    imageContainer.innerHTML = '';

    // Limit the number of cascaded images to 5 or less
    const cascadeLimit = Math.min(currentIndex, 10);
    const start = Math.max(0, currentIndex - cascadeLimit);

    // Loop through up to 5 previous images and append them in a cascade
    for (let i = start; i <= currentIndex; i++) {
        const img = loadImage(diffusionImages[i] || placeholderImage);

        // Main image (centered and fully visible, no transparency)
        if (i === currentIndex) {
            img.classList.add('current-image');
            img.style.opacity = '1'; // Main image has 100% opacity
        }
        // Cascaded images
        else {
            img.style.transform = `translate(-${(currentIndex - i) * 15}px, -${(currentIndex - i) * 15}px)`;
            img.style.opacity = '0.5'; // Cascading images have 50% opacity
        }

        imageContainer.appendChild(img); // Add image to container
    }
}

// Play button functionality
const playButton = document.getElementById('playButton');
let playing = false;

playButton.addEventListener('click', async () => {
    if (playing) {
        // Implement pause functionality if needed
        playing = false;
        playButton.textContent = "Diffuse !";
    } else {
        playing = true;
        playButton.textContent = "Generating...";
        playButton.disabled = true;
        const labelValue = parseInt(labelSlider.value);
        const startStep = parseInt(diffusionSlider.value);
        await runSampling(labelValue, startStep, 50, 2.0); // Start from current slider value
        playing = false;
        playButton.textContent = "Diffuse !";
        playButton.disabled = false;
    }
});

let session;

async function loadModel() {
    session = await ort.InferenceSession.create('./model.onnx');
    const loadingText = document.getElementById("loadingText");
    const spinner = document.querySelector(".spinner");
    // setTimeout(() => {
    //     loadingText.textContent = "INFO";
    //     spinner.style.display = "none";  // Hide spinner after loading
    // }, 3000); // Simulate a 3-second loading time
    loadingText.textContent = "INFO";
    spinner.style.display = "none";  // Hide spinner after loading
    console.log("Model loaded");
}

// Call loadModel when the page loads
loadModel();

// Helper function to create a noise tensor (zt)
function createNoiseTensor(batchSize, channels, height, width) {
    const totalSize = batchSize * channels * height * width;
    const noiseArray = new Float32Array(totalSize);

    // Fill with random noise between -1 and 1
    for (let i = 0; i < totalSize; i++) {
        noiseArray[i] = Math.random() * 2 - 1;
    }

    return new ort.Tensor('float32', noiseArray, [batchSize, channels, height, width]);
}

// Helper function to create a time tensor (t)
function createTimeTensor(batchSize, tValue) {
    const timeArray = new Float32Array(batchSize).fill(tValue);
    return new ort.Tensor('float32', timeArray, [batchSize]);
}

// Function to update zt: zt = zt - dt * vc
function updateZt(zt, vc, dt) {
    const ztData = zt.data;
    const vcData = vc.data;
    const newData = new Float32Array(ztData.length);
    for (let i = 0; i < ztData.length; i++) {
        newData[i] = ztData[i] - dt * vcData[i];
    }
    return new ort.Tensor('float32', newData, zt.dims);
}

// Function to compute guided vc: vc = vu + cfg * (vc - vu)
function computeGuidedVc(vc, vu, cfg) {
    const vcData = vc.data;
    const vuData = vu.data;
    const newData = new Float32Array(vcData.length);
    for (let i = 0; i < vcData.length; i++) {
        newData[i] = vuData[i] + cfg * (vcData[i] - vuData[i]);
    }
    return new ort.Tensor('float32', newData, vc.dims);
}

// Function to clone a tensor
function cloneTensor(tensor) {
    const clonedData = tensor.data.slice(); // Copy the data
    const clonedDims = tensor.dims.slice(); // Copy the dimensions
    return new ort.Tensor(tensor.type, clonedData, clonedDims);
}

// Function to run the sampling process
async function runSampling(conditionIndex, start = 0, steps = 50, cfg = 2.0) {
    const batchSize = 1; // Batch size of 1 for demo
    const channels = 1; // Updated to 1 channel for grayscale
    const height = 32;
    const width = 32;

    let zt;

    if (start === 0) {
        // Initialize noise (zt)
        zt = createNoiseTensor(batchSize, channels, height, width);
        diffusionzt[0] = cloneTensor(zt);
    } else if (diffusionzt[start]) {
        // Retrieve zt from previous computation
        zt = cloneTensor(diffusionzt[start]);
    } else {
        // Recompute zt up to the start step using previous conditioning (assumed to be 0)
        zt = createNoiseTensor(batchSize, channels, height, width);
        const dt = 1.0 / steps; // Time step delta

        for (let i = 0; i < start; i++) {
            const tValue = 1 - i / steps; // Adjusted to go from 1 to 0
            const t = createTimeTensor(batchSize, tValue); // Time step

            // Create conditioning label as int64 tensor with previous conditioning (0)
            const condArray = BigInt64Array.from([BigInt(0)]);
            const cond = new ort.Tensor('int64', condArray, [batchSize]);

            // First inference with 'cond' to get 'vc'
            let feeds = {
                zt: zt,
                t: t,
                cond: cond
            };

            const outputVc = await session.run(feeds);
            const vc = outputVc.output; // 'output' is the name of the output tensor

            // Second inference with 'null_cond' to get 'vu'
            const nullCondArray = BigInt64Array.from([BigInt(10)]); // Assuming null_cond is 10
            const nullCond = new ort.Tensor('int64', nullCondArray, [batchSize]);

            feeds.cond = nullCond;
            const outputVu = await session.run(feeds);
            const vu = outputVu.output;

            // Compute guided vc: vc = vu + cfg * (vc - vu)
            const guidedVc = computeGuidedVc(vc, vu, cfg);

            // Update zt: zt = zt - dt * vc
            zt = updateZt(zt, guidedVc, dt);

            // Store zt at each step
            diffusionzt[i + 1] = cloneTensor(zt);

            // Optionally, display images for steps leading up to the start
            displayImage(zt, height, width, i + 1);
        }
    }

    const dt = 1.0 / steps; // Time step delta

    // Perform sampling loop from the start step to the end
    for (let i = start; i < steps; i++) {
        const tValue = 1 - i / steps; // Adjusted to go from 1 to 0
        const t = createTimeTensor(batchSize, tValue); // Time step

        // Create conditioning label as int64 tensor
        const condArray = BigInt64Array.from([BigInt(conditionIndex)]);
        const cond = new ort.Tensor('int64', condArray, [batchSize]);

        // First inference with 'cond' to get 'vc'
        let feeds = {
            zt: zt,
            t: t,
            cond: cond
        };

        const outputVc = await session.run(feeds);
        const vc = outputVc.output; // 'output' is the name of the output tensor

        // Second inference with 'null_cond' to get 'vu'
        const nullCondArray = BigInt64Array.from([BigInt(10)]); // Assuming null_cond is 10
        const nullCond = new ort.Tensor('int64', nullCondArray, [batchSize]);

        feeds.cond = nullCond;
        const outputVu = await session.run(feeds);
        const vu = outputVu.output;

        // Compute guided vc: vc = vu + cfg * (vc - vu)
        const guidedVc = computeGuidedVc(vc, vu, cfg);

        // Update zt: zt = zt - dt * vc
        zt = updateZt(zt, guidedVc, dt);

        // Store zt at this step
        diffusionzt[i + 1] = cloneTensor(zt);

        // Display the current image
        displayImage(zt, height, width, i + 1);

        // Wait for a short duration to visualize each step
        await new Promise(resolve => setTimeout(resolve, 10)); // Adjust delay as needed
    }
}

// Function to display the generated image
function displayImage(tensor, height, width, index) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const imgData = ctx.createImageData(width, height);

    // Extract data
    const outputArray = tensor.data;

    // Rearrange the data to [height, width]
    for (let h = 0; h < height; h++) {
        for (let w = 0; w < width; w++) {
            const idx = h * width + w;

            let value = outputArray[idx];

            // Unnormalize and clamp to [0, 255]
            value = Math.floor(Math.min(Math.max((value * 0.5 + 0.5) * 255, 0), 255));

            // Set pixel data (R, G, B are the same for grayscale)
            const pixelIndex = (h * width + w) * 4;
            imgData.data[pixelIndex] = value;       // R
            imgData.data[pixelIndex + 1] = value;   // G
            imgData.data[pixelIndex + 2] = value;   // B
            imgData.data[pixelIndex + 3] = 255;     // A (fully opaque)
        }
    }

    ctx.imageSmoothingEnabled = false;
    ctx.putImageData(imgData, 0, 0);

    // Get data URL and store it in diffusionImages
    const dataURL = canvas.toDataURL();
    diffusionImages[index] = dataURL;

    // Update the slider and display the image
    diffusionSlider.value = index;
    sliderValueDisplay.textContent = index;
    updateImageDisplay(index);
}

// Initialize the image display
updateImageDisplay(parseInt(diffusionSlider.value));
