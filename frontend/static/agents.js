const token = localStorage.getItem("access_token");
const logoutButton = document.querySelector("#logout");

if (!token) {
  window.location.href = "/";
}

logoutButton.addEventListener("click", () => {
  localStorage.removeItem("access_token");
  localStorage.removeItem("current_user");
  window.location.href = "/";
});
