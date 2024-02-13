// Global messages array to store chat history
let messages = [];

document.addEventListener('DOMContentLoaded', function() {
    initializeChat();
});

function initializeChat() {
    loadMessages();
    setupMessageSubmission();
}

function setupMessageSubmission() {
    const submitButton = document.getElementById('submit');
    const inputElement = document.getElementById('inputText');

    submitButton.addEventListener('click', () => {
        const inputText = inputElement.value.trim();
        if (inputText) {
            sendUserMessage(inputText);
            inputElement.value = ''; // Clear input field after sending the message
        }
    });
}

function sendUserMessage(text) {
    addMessage(text, 'user');
    fetchArticleContentAndSendMessage(text);
}

function addMessage(text, sender) {
    const message = { text, sender };
    messages.push(message);
    displayMessage(message);
    saveMessages();
}

function displayMessage({ text, sender }) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    messageDiv.innerHTML = getMessageHTML(text, sender);
    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function getMessageHTML(text, sender) {
    if (sender === 'server') {
        return `
            <div class="message-content">
                <div class="message-icon" style="background-image: url('${chrome.runtime.getURL("images/icon48.png")}');"></div>
                <div class="message-text">${text}</div>
            </div>
        `;
    } else {
        return text;
    }
}

function saveMessages() {
    chrome.storage.local.set({'chatHistory': messages}, function() {
        logError('saving');
    });
}

function loadMessages() {
    chrome.storage.local.get('chatHistory', function(data) {
        if (!chrome.runtime.lastError && data.chatHistory) {
            messages = data.chatHistory;
            messages.forEach(displayMessage);
        } else {
            logError('loading');
        }
    });
}

function setupMessageSubmission() {
    const inputElement = document.getElementById('inputText');

    // Event listener for the button
    const submitButton = document.getElementById('submit');
    submitButton.addEventListener('click', () => {
        const inputText = inputElement.value.trim();
        if (inputText) {
            sendUserMessage(inputText);
            inputElement.value = ''; // Clear input field after sending the message
        }
    });

    // Event listener for the Enter key
    inputElement.addEventListener('keypress', (event) => {
        if (event.key === 'Enter') { // 13 is the Enter key
            event.preventDefault(); // Prevent the default action to stop it from submitting a form if any
            submitButton.click(); // Trigger the button click programmatically
        }
    });
}

function fetchArticleContentAndSendMessage(inputText) {
    chrome.runtime.sendMessage({action: "fetchArticleContent", content: inputText}, function(response) {
        addMessage(response.result, 'server');
    });
}

function logError(context) {
    if (chrome.runtime.lastError) {
        console.error(`Error ${context} to storage: ${chrome.runtime.lastError}`);
    } else if (context === 'saving') {
        console.log('Chat history saved.');
    }
}