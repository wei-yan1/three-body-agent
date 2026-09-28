const form = document.querySelector("#auth-form");
const switchMode = document.querySelector("#switch-mode");
const submitButton = document.querySelector("#submit-button");
const message = document.querySelector("#message");
const confirmField = document.querySelector("#confirm-field");
let mode = "login";

function setMessage(text, type = "") { message.textContent = text; message.className = `message ${type}`; }
function renderMode() {
  const register = mode === "register";
  document.querySelector("#auth-kicker").textContent = register ? "第一次来到这里" : "欢迎回来";
  document.querySelector("#auth-title").textContent = register ? "建立你的档案室" : "进入你的档案室";
  document.querySelector("#auth-description").textContent = register ? "创建账号，保存属于你的小说和角色。" : "继续上次留下的小说与角色。";
  submitButton.querySelector("span").textContent = register ? "创建档案室" : "进入档案室";
  switchMode.textContent = register ? "已经有账号？返回登录" : "还没有账号？创建一个";
  confirmField.classList.toggle("hidden", !register);
  const password = document.querySelector("#password");
  password.autocomplete = register ? "new-password" : "current-password";
  password.minLength = register ? 8 : 1;
  password.placeholder = register ? "至少 8 位字符" : "输入密码";
  setMessage("");
}
switchMode.addEventListener("click", () => { mode = mode === "login" ? "register" : "login"; renderMode(); });
form.addEventListener("submit", async (event) => {
  event.preventDefault(); setMessage("");
  const username = document.querySelector("#username").value.trim();
  const password = document.querySelector("#password").value;
  if (username.length < 2) return setMessage("用户名至少需要 2 个字符。", "error");
  if (mode === "register" && password.length < 8) return setMessage("新密码至少需要 8 个字符。", "error");
  if (mode === "register" && password !== document.querySelector("#password-confirm").value) return setMessage("两次输入的密码不一致。", "error");
  setBusy(true);
  try {
    const response = await fetch(`/api/v1/auth/${mode}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username, password }) });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(Array.isArray(data.detail) ? "输入格式不正确。" : data.detail || "请求失败");
    localStorage.setItem("access_token", data.access_token); localStorage.setItem("current_user", JSON.stringify(data.user));
    setMessage(`欢迎，${data.user.display_name || data.user.username}`, "success"); window.setTimeout(() => { window.location.href = "/storyrole"; }, 250);
  } catch (error) { setMessage(error.message, "error"); setBusy(false); }
});
function setBusy(value) { submitButton.disabled = value; submitButton.querySelector("span").textContent = value ? "正在连接档案室…" : mode === "register" ? "创建档案室" : "进入档案室"; }
renderMode();
