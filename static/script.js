function toggleFeatures() {
    const box = document.getElementById("featureBox");

    if (box.style.display === "block") {
        box.style.display = "none";
    } else {
        box.style.display = "block";
    }
}

async function predict() {
    const input = document.getElementById("inputBox").value;
    const errorDiv = document.getElementById("error");
    const resultDiv = document.getElementById("result");

    errorDiv.innerText = "";
    resultDiv.innerText = "";

    let values = input.split(",").map(v => v.trim());

    if (values.length !== 30) {
        errorDiv.innerText = "Please enter exactly 30 values";
        return;
    }

    let numbers = values.map(v => parseFloat(v));

    if (numbers.some(isNaN)) {
        errorDiv.innerText = "Invalid number detected";
        return;
    }

    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {"Content-Type": "application/json"},
            body: JSON.stringify({features: numbers})
        });

        const data = await response.json();

        if (data.error) {
            errorDiv.innerText = data.error;
        } else {
            resultDiv.innerText = "Prediction: " + data.result;
        }

    } catch (err) {
        errorDiv.innerText = "Server error";
    }
}
