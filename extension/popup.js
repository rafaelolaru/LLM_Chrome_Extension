// Global messages array to store chat history
let messages = [];

document.addEventListener("DOMContentLoaded", function () {
  loadMessages();
  const submitButton = document.getElementById("submit");
  const inputElement = document.getElementById("inputText");
  const clearMessagesButton = document.getElementById("clearMessages");

  // Setup message submission on 'Send' button click
  submitButton.addEventListener("click", () => {
    const inputText = inputElement.value.trim();
    if (inputText) {
      sendUserMessage(inputText);
      inputElement.value = ""; // Clear input field after sending the message
    }
  });

  // Setup message submission on 'Enter' key press
  inputElement.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent the default action to stop it from submitting a form if any
      submitButton.click(); // Trigger the button click programmatically
    }
  });

  // Setup auto-scroll on new input
  inputElement.addEventListener("input", () => {
    const chatArea = document.getElementById("chatArea");
    chatArea.scrollTop = chatArea.scrollHeight;
  });

  // Setup clear messages on 'Clear Messages' button click
  clearMessagesButton.addEventListener("click", function () {
    clearMessages(); // Clear messages locally
    fetch(
      "https://62fa-2a02-2f09-300f-ca00-bd6d-f9de-3152-e064.ngrok-free.app/clear_memory",
      { method: "DELETE" }
    ) // Call the endpoint to clear server memory
      .then((response) => response.json())
      .then((data) => console.log(data.status))
      .catch((error) => console.error("Error clearing memory:", error));
  });
});

function autoScrollChat() {
  const chatArea = document.getElementById("chatArea");
  chatArea.scrollTop = chatArea.scrollHeight;
}

function sendUserMessage(text) {
  addMessage(text, "user");
  fetchArticleContentAndSendMessage(text);
}

function addMessage(text, sender) {
  const message = { text, sender };
  messages.push(message);
  displayMessage(message);
  saveMessages();
}

function displayMessage({ text, sender }) {
  const chatArea = document.getElementById("chatArea");
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}`;
  messageDiv.innerHTML = getMessageHTML(text, sender);
  chatArea.appendChild(messageDiv);
  autoScrollChat();
}

function getMessageHTML(text, sender) {
  if (sender === "server") {
    return `
            <div class="message-content">
                <div class="message-icon" style="background-image: url('${chrome.runtime.getURL(
                  "images/icon48.png"
                )}');"></div>
                <div class="message-text">${text}</div>
            </div>
        `;
  } else {
    return text;
  }
}

function saveMessages() {
  chrome.storage.local.set({ chatHistory: messages }, function () {
    if (chrome.runtime.lastError) {
      console.error("Error saving messages:", chrome.runtime.lastError);
    } else {
      console.log("Chat history saved.");
    }
  });
}

function loadMessages() {
  chrome.storage.local.get("chatHistory", function (data) {
    if (!chrome.runtime.lastError && data.chatHistory) {
      messages = data.chatHistory;
      messages.forEach(displayMessage);
    } else {
      logError("");
    }
  });
}

function setupEnterKeyListener() {
  const inputElement = document.getElementById("inputText");
  inputElement.addEventListener("keypress", (event) => {
    if (event.key === "Enter") {
      event.preventDefault(); // Prevent the default action
      document.getElementById("submit").click(); // Trigger the button click programmatically
    }
  });
}

function fetchArticleContentAndSendMessage(inputText) {
  chrome.runtime.sendMessage(
    { action: "fetchArticleContent", content: inputText },
    function (response) {
      addMessage(response.result, "server");
    }
  );
}

function clearMessages() {
  messages = []; // Clear the local messages array
  chrome.storage.local.remove("chatHistory", function () {
    if (!chrome.runtime.lastError) {
      console.log("Chat history cleared from storage.");
      const chatArea = document.getElementById("chatArea");
      chatArea.innerHTML = ""; // Clear the chat area UI
    } else {
      console.error(
        "Error clearing chat history from storage:",
        chrome.runtime.lastError
      );
    }
  });
}
