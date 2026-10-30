const form = document.getElementById("upload-form");
const input = document.getElementById("image-input");
const previewWrap = document.getElementById("preview-wrap");
const preview = document.getElementById("preview");
const clearButton = document.getElementById("clear-button");
const submitButton = document.getElementById("submit-button");
const status = document.getElementById("status");

const result = document.getElementById("result");
const prediction = document.getElementById("prediction");
const confidence = document.getElementById("confidence");
const probabilities = document.getElementById("probabilities");

input.addEventListener("change", () => {
    const file = input.files[0];

    if (!file) {
        resetPreview();
        return;
    }

    preview.src = URL.createObjectURL(file);
    previewWrap.classList.remove("hidden");
    submitButton.disabled = false;
    status.textContent = "";
    result.classList.add("hidden");
});

clearButton.addEventListener("click", () => {
    input.value = "";
    resetPreview();
});

form.addEventListener("submit", async (event) => {
    event.preventDefault();

    const file = input.files[0];
    if (!file) return;

    submitButton.disabled = true;
    submitButton.textContent = "Analyzing…";
    status.textContent = "Running the model…";
    result.classList.add("hidden");

    const formData = new FormData();
    formData.append("image", file);

    try {
        const response = await fetch("/predict", {
            method: "POST",
            body: formData,
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed.");
        }

        prediction.textContent = data.prediction;
        confidence.textContent = `${data.confidence.toFixed(2)}%`;

        probabilities.innerHTML = data.results.map((item) => `
            <div class="probability-row">
                <span class="probability-name">${item.label}</span>
                <div class="bar">
                    <div
                        class="bar-fill"
                        style="width: ${Math.min(item.probability, 100)}%"
                    ></div>
                </div>
                <span class="probability-value">${item.probability.toFixed(2)}%</span>
            </div>
        `).join("");

        result.classList.remove("hidden");
        status.textContent = "";
    } catch (error) {
        status.textContent = error.message;
    } finally {
        submitButton.disabled = false;
        submitButton.textContent = "Analyze image";
    }
});

function resetPreview() {
    previewWrap.classList.add("hidden");
    preview.removeAttribute("src");
    submitButton.disabled = true;
    status.textContent = "";
    result.classList.add("hidden");
}
