document.addEventListener("DOMContentLoaded", function () {
    // Get the user's geolocation
    navigator.geolocation.getCurrentPosition(function (position) {
        document.getElementById('latitude').value = position.coords.latitude;
        document.getElementById('longitude').value = position.coords.longitude;
    });

    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
        const recognition = new SpeechRecognition();
        recognition.continuous = true;
        recognition.interimResults = true;

        const startBtn = document.getElementById('start-record-btn');
        const stopBtn = document.getElementById('stop-record-btn');
        const searchInput = document.getElementById('search');
        const form = document.getElementById('report-form');
        const loader = document.getElementById('loader');
        const loaderText = document.getElementById('loader-text');
        const progressBarFill = document.getElementById('progress-bar-fill');
        const submitBtn = document.getElementById('submit-btn');

        let finalTranscript = "";

        startBtn.addEventListener('click', () => {
            recognition.start();
            startBtn.style.display = 'none';
            stopBtn.style.display = 'inline-block';
        });

        stopBtn.addEventListener('click', () => {
            recognition.stop();
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        });

        recognition.addEventListener('result', (event) => {
            let interimTranscript = "";
            for (let i = event.resultIndex; i < event.results.length; i++) {
                if (event.results[i].isFinal) {
                    finalTranscript += event.results[i][0].transcript + " ";
                } else {
                    interimTranscript += event.results[i][0].transcript;
                }
            }
            searchInput.value = finalTranscript + interimTranscript;
        });

        recognition.addEventListener('end', () => {
            startBtn.style.display = 'inline-block';
            stopBtn.style.display = 'none';
        });

        form.addEventListener('submit', (e) => {
            e.preventDefault();
            loader.style.display = 'block';
            progressBarFill.style.width = '50%';
            loaderText.innerText = 'Uploading...';

            setTimeout(() => {
                progressBarFill.style.width = '100%';
                loaderText.innerText = 'Submitted!';
                form.reset();
                loader.style.display = 'none';
            }, 3000);
        });
    } else {
        alert("Your browser does not support speech recognition.");
    }
});