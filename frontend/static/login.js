const form = document.querySelector("#auth-form");
const modeToggle = document.querySelector("#mode-toggle");
const switchMode = document.querySelector("#switch-mode");
const message = document.querySelector("#message");
let mode = "login";

modeToggle.addEventListener("click", () => {
  form.classList.toggle("visible");
});

switchMode.addEventListener("click", () => {
  mode = mode === "login" ? "register" : "login";
  modeToggle.textContent = mode === "login" ? "log in" : "sign up";
  switchMode.textContent = mode === "login" ? "create account" : "back to login";
  message.textContent = "";
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  message.textContent = "";
  const payload = {
    username: document.querySelector("#username").value.trim(),
    password: document.querySelector("#password").value,
  };
  const endpoint = mode === "login" ? "/api/v1/auth/login" : "/api/v1/auth/register";

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "request failed");
    }
    localStorage.setItem("access_token", data.access_token);
    localStorage.setItem("current_user", JSON.stringify(data.user));
    message.textContent = `welcome, ${data.user.username}`;
    window.location.href = "/agents";
  } catch (error) {
    message.textContent = error.message;
  }
});
