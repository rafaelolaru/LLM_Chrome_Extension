// register.js
document.addEventListener("DOMContentLoaded", function () {
  const registerButton = document.getElementById("registerButton");

  registerButton.addEventListener("click", () => {
    const email = document.getElementById("email").value;
    // Assume you have a username field, since it's referenced in the JSON body.
    const username = document.getElementById("username").value;
    const password = document.getElementById("password").value;
    const confirmPassword = document.getElementById("confirmPassword").value;

    if (password === confirmPassword) {
      // Passwords match, proceed with registration
      fetch(
        "https://62fa-2a02-2f09-300f-ca00-bd6d-f9de-3152-e064.ngrok-free.app/register",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ email, username, password }),
        }
      )
        .then((response) => {
          if (!response.ok) {
            throw new Error("Network response was not ok");
          }
          return response.json();
        })
        .then((data) => {
          // Assuming 'data' contains some indication of a successful registration
          console.log(data);
          // Redirect the user to the login page upon successful registration
          window.location.href = "/login.html"; // Adjust the path as needed
        })
        .catch((error) => {
          // Handle errors
          console.error("Error:", error);
          alert("Registration failed. Please try again.");
        });
    } else {
      // Passwords do not match, show an error message
      alert("Passwords do not match.");
    }
  });
});
