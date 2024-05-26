document.getElementById('upload-form').addEventListener('submit', async function(event) {
    event.preventDefault();

    const fileInput = document.getElementById('audio-file');
    const file = fileInput.files[0];

    if (!file) {
        alert('Please select a file.');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:5000/emo', {
            method: 'POST',
            body: formData
        });
       

        if (!response.ok) {
            throw new Error('Network response was not ok');
        }

        const result = await response.json();
        displayResults(result);
        console.log(result);
        console.log("abc");
    } catch (error) {
        console.error('There was a problem with the fetch operation:', error);
    }
});

function displayResults(result) {
    const resultsDiv = document.getElementById('result');
    resultsDiv.innerHTML = '';

    const emotions = JSON.parse(result);
    emotions.forEach(emotion => {
        const emotionElement = document.createElement('div');
        emotionElement.textContent = `${emotion.time}: ${emotion.emotion}`;
        resultsDiv.appendChild(emotionElement);
    });
}
