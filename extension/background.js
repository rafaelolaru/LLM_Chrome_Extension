chrome.runtime.onMessage.addListener(
    function(request, sender, sendResponse) {
        if (request.action === "fetchArticleContent") {
            chrome.tabs.query({active: true, currentWindow: true}, function(tabs) {
                chrome.scripting.executeScript({
                    target: { tabId: tabs[0].id },
                    function: () => {
                        let articleContent = "";
                        const paragraphs = document.querySelectorAll('article p, main p');
                        paragraphs.forEach(p => {
                            const text = p.textContent.trim();
                            if (text.length > 100) {
                                articleContent += text + "\n\n";
                            }
                        });
                        return articleContent;
                    }
                }, (injectionResults) => {
                    const articleContent = injectionResults[0].result;
                    sendMessageToServer(request.content, articleContent, sendResponse);
                });
            });
            return true;
        }
    }
);

function sendMessageToServer(chatContent, articleContent, sendResponse) {
    fetch('http://127.0.0.1:5000/manipulate', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: chatContent, context: articleContent })
    })
    .then(response => response.json())
    .then(data => {
        sendResponse({ result: data.result });
    })
    .catch(error => {
        console.error('Error:', error);
        sendResponse({ result: 'Error occurred' });
    });
}
