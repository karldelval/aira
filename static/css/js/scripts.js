document.addEventListener("DOMContentLoaded", function() {
    const reportForm = document.getElementById("reportForm");
    const status = document.getElementById("status");

    // Request the user's location
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                document.getElementById("latitude").value = position.coords.latitude;
                document.getElementById("longitude").value = position.coords.longitude;
            },
            (error) => {
                status.textContent = "Location access denied. Please allow location access to use this feature.";
                console.error("Geolocation error:", error);
            }
        );
    } else {
        status.textContent = "Geolocation is not supported by your browser.";
    }

    // Form submission
    reportForm.addEventListener("submit", async (event) => {
        event.preventDefault();

        const formData = new FormData(reportForm);
        try {
            const response = await fetch("/report", {
                method: "POST",
                body: formData
            });
            const result = await response.json();
            status.textContent = result.message || "Report submitted successfully!";
            reportForm.reset();
        } catch (error) {
            status.textContent = "Error submitting report.";
            console.error("Submission error:", error);
        }
    });
});