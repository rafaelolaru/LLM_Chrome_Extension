chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request.action === "fetchArticleContent") {
    chrome.tabs.query({ active: true, currentWindow: true }, function (tabs) {
      chrome.scripting.executeScript(
        {
          target: { tabId: tabs[0].id },
          function: () => {
            let articleContent = "";
            const paragraphs = document.querySelectorAll("article p, main p");
            paragraphs.forEach((p) => {
              const text = p.textContent.trim();
              if (text.length > 100) {
                articleContent += text + "\n\n";
              }
            });
            return articleContent;
          },
        },
        (injectionResults) => {
          // Check if injectionResults is not null and has a result.
          let articleContent = "";
          if (
            injectionResults &&
            injectionResults[0] &&
            injectionResults[0].result
          ) {
            articleContent = injectionResults[0].result;
          }
          // Now articleContent will be an empty string if no content was retrieved,
          // ensuring sendMessageToServer is called with an empty content if needed.
          sendMessageToServer(request.content, articleContent, sendResponse);
        }
      );
    });
    // Indicate that the response is asynchronous.
    return true;
  }
});

function sendMessageToServer(chatContent, articleContent, sendResponse) {
  // Retrieve the access token from chrome.storage.sync
  chrome.storage.sync.get("access_token", function (data) {
    const token = data.access_token;
    if (!token) {
      console.error("No access token available.");
      sendResponse({ result: "No access token available." });
      return;
    }

    // Now, with the token available, make the fetch request
    fetch("https://rafaelolaru.xyz/manipulate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ query: chatContent, context: articleContent }),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        sendResponse({ result: data.result });
      })
      .catch((error) => {
        console.error("Error:", error);
        sendResponse({ result: "Error occurred" });
      });
  });

  return true;
}
