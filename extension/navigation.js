// This script will be responsible for navigation-related tasks

document.addEventListener("DOMContentLoaded", () => {
  const registerBtn = document.getElementById("goToRegister");
  if (registerBtn) {
    registerBtn.addEventListener("click", () => {
      window.location.href = "register.html";
    });
  }
});
