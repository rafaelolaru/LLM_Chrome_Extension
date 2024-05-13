// login.js

document.addEventListener("DOMContentLoaded", () => {
  chrome.storage.sync.get("access_token", (items) => {
    const token = items.access_token;
    if (token) {
      // Call the verify_token endpoint to check token validity
      fetch("https://rafaelolaru.xyz/verify_token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      })
        .then((response) => response.json())
        .then((data) => {
          if (data.valid) {
            // If the token is valid, redirect to popup.html
            window.location.href = "popup.html";
          } else {
            // If the token is invalid or expired, stay on login.html
            console.log("Token is invalid or expired");
            // Optionally, clear the invalid token
            chrome.storage.sync.remove("access_token");
          }
        })
        .catch((error) => {
          console.error("Error verifying token:", error);
        });
    }
    // If no token is found, stay on login.html for the user to login
  });
});

document.addEventListener("DOMContentLoaded", function () {
  const loginButton = document.getElementById("loginButton");

  loginButton.addEventListener("click", () => {
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;

    // Proceed with login
    fetch("https://rafaelolaru.xyz/authenticate", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ username, password }),
    })
      .then((response) => {
        if (!response.ok) {
          // This throws an error and skips to the catch block
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        // Handle response data
        console.log(data);
        // Store the access token for future requests
        chrome.storage.sync.set({ access_token: data.access_token }, () => {
          console.log("Access token stored.");
        });
        // Redirect the user to popup.html upon successful login
        window.location.href = "popup.html";
      })
      .catch((error) => {
        // Handle errors
        console.error("Error:", error);
        alert(
          "Login failed. Please check your username and password and try again."
        );
      });
  });
});
