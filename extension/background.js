chrome.runtime.onMessage.addListener(
  function(request, sender, sendResponse) {
      fetch('http://192.168.1.128:5000/manipulate', {
          method: 'POST',
          headers: {
              'Content-Type': 'application/json',
          },
          body: JSON.stringify({ text: request.content })
      })
      .then(response => response.json())
      .then(data => {
          sendResponse(data.result);
      })
      .catch(error => {
          console.error('Error:', error);
          sendResponse('Error occurred');
      });
      return true; // Keep the message channel open for the response
  }
);
