let messages = []; // This should be outside of the event listener to maintain scope across function calls

function addMessage(text, sender) {
    // Create message object
    const message = { text: text, sender: sender };
    messages.push(message);
    appendMessageToChatArea(message);
    saveMessages();
}

function appendMessageToChatArea(message) {
    const chatArea = document.getElementById('chatArea');
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', message.sender);

    if (message.sender === 'server') {
        const messageContentDiv = document.createElement('div');
        messageContentDiv.classList.add('message-content');

        const iconDiv = document.createElement('div');
        iconDiv.classList.add('message-icon');
        // Set the background image dynamically using chrome.runtime.getURL
        iconDiv.style.backgroundImage = `url('${chrome.runtime.getURL("images/icon48.png")}')`;

        const textDiv = document.createElement('div');
        textDiv.classList.add('message-text');
        textDiv.textContent = message.text; // Ensure any user-generated content is safely handled

        messageContentDiv.appendChild(iconDiv);
        messageContentDiv.appendChild(textDiv);
        messageDiv.appendChild(messageContentDiv);
    } else { // User message
        messageDiv.textContent = message.text;
    }

    chatArea.appendChild(messageDiv);
    chatArea.scrollTop = chatArea.scrollHeight; // Scroll to the bottom
}
function saveMessages() {
    chrome.storage.local.set({ 'chatHistory': messages }, function() {
        if (chrome.runtime.lastError) {
            console.error(`Error saving to storage: ${chrome.runtime.lastError}`);
        } else {
            console.log('Chat history saved.');
        }
    });
}

// Loading data from storage
function loadMessages() {
    chrome.storage.local.get('chatHistory', function(data) {
        if (chrome.runtime.lastError) {
            console.error(`Error loading from storage: ${chrome.runtime.lastError}`);
        } else if (data.chatHistory && data.chatHistory.length > 0) {
            messages = data.chatHistory;
            messages.forEach(message => appendMessageToChatArea(message));
        }
    });
}

document.addEventListener('DOMContentLoaded', function() {
    loadMessages();
    const submitButton = document.getElementById('submit');
    const inputElement = document.getElementById('inputText'); // Get input element once at the beginning

    submitButton.addEventListener('click', function() {
        const inputText = inputElement.value;
        addMessage(inputText, 'user');
        inputElement.value = ''; // Clear input field after adding message

        chrome.runtime.sendMessage({ content: inputText }, function(response) {
            addMessage(response, 'server');
        });
    });
});




// document.addEventListener('DOMContentLoaded', function() {
//     const submitButton = document.getElementById('submit');
//     const chatArea = document.getElementById('chatArea');
//     let messages = [];
//     submitButton.addEventListener('click', function() {
//         const inputText = document.getElementById('inputText').value;
//         addMessage(inputText, 'user');

//         chrome.runtime.sendMessage({ content: inputText }, function(response) {
//             addMessage(response, 'server');
//         });
//     });

//     function addMessage(text, sender) {
//         const messageDiv = document.createElement('div');
//         messageDiv.classList.add('message', sender);
    
//         if (sender === 'server') {
//             const messageContentDiv = document.createElement('div');
//             messageContentDiv.classList.add('message-content');
    
//             const iconDiv = document.createElement('div');
//             iconDiv.classList.add('message-icon');
            
//             // Set the background image dynamically using chrome.runtime.getURL
//             iconDiv.style.backgroundImage = `url('${chrome.runtime.getURL("images/icon48.png")}')`;
    
//             const textDiv = document.createElement('div');
//             textDiv.classList.add('message-text');
//             textDiv.textContent = text;
    
//             messageContentDiv.appendChild(iconDiv);
//             messageContentDiv.appendChild(textDiv);
//             messageDiv.appendChild(messageContentDiv);
//         } else {
//             messageDiv.textContent = text;
//         }
    
//         chatArea.appendChild(messageDiv);
//         document.getElementById('inputText').value = ''; // Clear input field
//         chatArea.scrollTop = chatArea.scrollHeight; // Scroll to the bottom
//     }
// });
