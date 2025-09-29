let model;

async function loadModel() {
  if (!model) {
    try {
      model = await tf.loadLayersModel('model/model.json');
      console.log("Model loaded successfully");
    } catch (err) {
      console.error("Failed to load model:", err);
    }
  }
}

function preprocessImage(img, size = 224) {
  // Create a canvas to resize the image
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');

  // Draw the image onto the canvas and resize
  ctx.drawImage(img, 0, 0, size, size);

  // Get image data and convert to tensor
  const imageData = ctx.getImageData(0, 0, size, size);
  let tensor = tf.browser.fromPixels(imageData)
                .toFloat()
                .div(tf.scalar(255))
                .expandDims(); // Add batch dimension

  return tensor;
}

async function analyzeImages() {
  await loadModel();

  const files = document.getElementById("imageUpload").files;
  const resultsDiv = document.getElementById("results");
  resultsDiv.innerHTML = "";

  if (files.length === 0) {
    resultsDiv.innerHTML = "<p>Please upload at least one image.</p>";
    return;
  }

  for (let file of files) {
    const img = document.createElement("img");
    img.src = URL.createObjectURL(file);

    img.onload = async () => {
      try {
        const tensor = preprocessImage(img);
        const prediction = model.predict(tensor);
        const probs = await prediction.data();
        const labels = ["Normal Eye", "Possible retinoblastoma/cataract"];

        const card = document.createElement("div");
        card.className = "prediction-card";
        card.appendChild(img);

        labels.forEach((label, i) => {
          const percent = (probs[i] * 100).toFixed(2);

          const barContainer = document.createElement("div");
          barContainer.className = "bar-container";

          const barLabel = document.createElement("span");
          barLabel.textContent = `${label}: ${percent}%`;

          const bar = document.createElement("div");
          bar.className = "bar";
          bar.style.width = "0%";
          if (label.toLowerCase().includes("retinoblastoma") && percent > 50) {
            bar.style.background = "red";
          }

          barContainer.appendChild(barLabel);
          barContainer.appendChild(bar);
          card.appendChild(barContainer);

          setTimeout(() => {
            bar.style.width = `${percent}%`;
          }, 200);
        });

        resultsDiv.appendChild(card);

        tensor.dispose(); // free memory
      } catch (err) {
        console.error("Prediction failed for", file.name, err);
      }
    };

    img.onerror = () => {
      console.error("Failed to load image:", file.name);
    };
  }
}
