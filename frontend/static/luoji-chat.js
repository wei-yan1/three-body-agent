const token = localStorage.getItem("access_token");
const agentSlug = document.body.dataset.agent || "luoji";
const agentName = document.body.dataset.agentName || "罗辑";
const apiBase = `/api/v1/chat/${agentSlug}`;

const form = document.querySelector("#chat-form");
const messageInput = document.querySelector("#user-message");
const messages = document.querySelector("#messages");
const sendButton = document.querySelector("#send-button");
const stagePicker = document.querySelector("#stage-picker");
const stageTrigger = document.querySelector("#stage-trigger");
const stageOptions = Array.from(document.querySelectorAll(".stage-option"));
const threadPicker = document.querySelector("#thread-picker");
const threadTrigger = document.querySelector("#thread-trigger");
const threadOptions = document.querySelector("#thread-options");
const threadSelectedCount = document.querySelector("#thread-selected-count");
const deleteThreadsButton = document.querySelector("#delete-threads");
const newThreadButton = document.querySelector("#new-thread");
const customThreadButton = document.querySelector("#custom-thread");
const tooltip = document.querySelector("#tooltip");

let knownThreads = [{ id: null, thread_name: "线程1" }];
let markedThreadIds = new Set();

if (!token) {
  window.location.href = "/";
}

function selectedStage() {
  return stagePicker.dataset.value;
}

function selectedMode() {
  return document.querySelector('input[name="knowledge-mode"]:checked').value;
}

function selectedThreadName() {
  return threadPicker.dataset.value || "线程1";
}

function selectedThreadId() {
  return threadPicker.dataset.threadId ? Number(threadPicker.dataset.threadId) : null;
}

function setThreadName(name, id = null) {
  const normalized = (name || "").trim() || "线程1";
  threadPicker.dataset.value = normalized;
  threadPicker.dataset.threadId = id ? String(id) : "";
  threadTrigger.textContent = normalized;
  renderThreadOptions();
}

function nextThreadName() {
  const used = new Set(knownThreads.map((thread) => thread.thread_name));
  let index = 1;
  while (used.has(`线程${index}`)) {
    index += 1;
  }
  return `线程${index}`;
}

function renderThreadOptions() {
  threadOptions.innerHTML = "";
  const selected = selectedThreadName();
  const threadMap = new Map();
  threadMap.set("线程1", { id: null, thread_name: "线程1" });
  knownThreads.forEach((thread) => threadMap.set(thread.thread_name, thread));
  threadMap.set(selected, { id: selectedThreadId(), thread_name: selected });
  knownThreads = Array.from(threadMap.values());

  knownThreads.forEach((thread) => {
    const button = document.createElement("button");
    const isMarked = thread.id && markedThreadIds.has(Number(thread.id));
    button.className = [
      "thread-option",
      thread.thread_name === selected ? "active" : "",
      isMarked ? "marked" : "",
      thread.id ? "" : "placeholder",
    ].filter(Boolean).join(" ");
    button.type = "button";
    button.dataset.threadId = thread.id || "";

    const nameSpan = document.createElement("span");
    nameSpan.className = "thread-name";
    nameSpan.textContent = thread.thread_name;

    const selector = document.createElement("span");
    selector.className = "thread-select-zone";
    selector.title = thread.id ? "选择后可批量删除" : "线程未保存，发送第一句话后才可删除";
    selector.setAttribute("role", "checkbox");
    selector.setAttribute("aria-label", thread.id ? `选择删除 ${thread.thread_name}` : `${thread.thread_name} 尚未保存`);
    selector.setAttribute("aria-checked", String(Boolean(isMarked)));
    selector.setAttribute("aria-disabled", String(!thread.id));

    const dot = document.createElement("span");
    dot.className = "thread-select-dot";
    selector.appendChild(dot);
    button.append(nameSpan, selector);

    selector.addEventListener("click", (event) => {
      event.stopPropagation();
      toggleThreadMarked(thread.id);
    });

    button.addEventListener("click", (event) => {
      if (selector.contains(event.target)) {
        return;
      }
      setThreadName(thread.thread_name, thread.id);
      closeThreadMenu();
      loadThreadMessages(thread.id);
    });

    threadOptions.appendChild(button);
  });

  newThreadButton.textContent = `新建${nextThreadName()}`;
  updateDeleteButton();
}

function toggleThreadMarked(threadId) {
  if (!threadId) {
    return;
  }
  const id = Number(threadId);
  if (markedThreadIds.has(id)) {
    markedThreadIds.delete(id);
  } else {
    markedThreadIds.add(id);
  }
  renderThreadOptions();
}

function updateDeleteButton() {
  const count = markedThreadIds.size;
  deleteThreadsButton.disabled = count === 0;
  deleteThreadsButton.textContent = count > 0 ? `批量删除选中线程（${count}）` : "批量删除选中线程";
  if (threadSelectedCount) {
    threadSelectedCount.textContent = `${count} 已选`;
  }
}

async function loadThreads() {
  try {
    const params = new URLSearchParams({
      timeline_stage: selectedStage(),
      knowledge_mode: selectedMode(),
    });
    const response = await fetch(`${apiBase}/threads?${params.toString()}`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      throw new Error("线程列表读取失败");
    }
    const data = await response.json();
    knownThreads = data.length > 0 ? data : [{ id: null, thread_name: "线程1" }];
    const existingIds = new Set(knownThreads.filter((thread) => thread.id).map((thread) => Number(thread.id)));
    markedThreadIds = new Set([...markedThreadIds].filter((id) => existingIds.has(id)));
    const first = knownThreads[0] || { id: null, thread_name: "线程1" };
    setThreadName(first.thread_name, first.id);
    loadThreadMessages(first.id);
  } catch {
    knownThreads = [{ id: null, thread_name: "线程1" }];
    setThreadName("线程1", null);
    clearMessages();
  }
}

async function loadThreadMessages(threadId) {
  clearMessages();
  if (!threadId) {
    return;
  }
  try {
    const response = await fetch(`${apiBase}/threads/${threadId}/messages`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    if (!response.ok) {
      return;
    }
    const data = await response.json();
    data.forEach((message) => appendMessage(message.role, message.content));
  } catch {
    clearMessages();
  }
}

function clearMessages() {
  messages.innerHTML = "";
}

function appendMessage(role, text) {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const paragraph = document.createElement("p");
  paragraph.textContent = text;
  article.appendChild(paragraph);
  messages.appendChild(article);
  messages.scrollTop = messages.scrollHeight;
}

function showTooltip(target) {
  const text = target.dataset.help;
  if (!text) {
    return;
  }
  tooltip.textContent = text;
  tooltip.classList.add("visible");
  moveTooltip(target);
}

function moveTooltip(target) {
  if (!tooltip.classList.contains("visible")) {
    return;
  }
  const rect = target.getBoundingClientRect();
  const gap = 10;
  const tooltipWidth = tooltip.offsetWidth || 320;
  let left = rect.right + gap;
  let top = rect.top + Math.max(0, (rect.height - tooltip.offsetHeight) / 2);

  if (left + tooltipWidth > window.innerWidth - 12) {
    left = rect.left;
    top = rect.bottom + gap;
  }
  tooltip.style.left = `${Math.max(12, left)}px`;
  tooltip.style.top = `${Math.max(12, top)}px`;
}

function hideTooltip() {
  tooltip.classList.remove("visible");
}

function bindTooltip(target) {
  target.addEventListener("mouseenter", () => showTooltip(target));
  target.addEventListener("mousemove", () => moveTooltip(target));
  target.addEventListener("mouseleave", hideTooltip);
  target.addEventListener("focus", () => showTooltip(target));
  target.addEventListener("blur", hideTooltip);
}

function closeStageMenu() {
  stagePicker.classList.remove("open");
  stageTrigger.setAttribute("aria-expanded", "false");
}

function closeThreadMenu() {
  threadPicker.classList.remove("open");
  threadTrigger.setAttribute("aria-expanded", "false");
}

stageTrigger.addEventListener("click", () => {
  closeThreadMenu();
  const isOpen = stagePicker.classList.toggle("open");
  stageTrigger.setAttribute("aria-expanded", String(isOpen));
});

threadTrigger.addEventListener("click", () => {
  closeStageMenu();
  const isOpen = threadPicker.classList.toggle("open");
  threadTrigger.setAttribute("aria-expanded", String(isOpen));
});

stageOptions.forEach((option) => {
  bindTooltip(option);
  option.addEventListener("click", () => {
    stagePicker.dataset.value = option.dataset.value;
    stageTrigger.textContent = option.textContent;
    stageTrigger.dataset.help = option.dataset.help;
    stageOptions.forEach((item) => item.classList.toggle("active", item === option));
    closeStageMenu();
    hideTooltip();
    loadThreads();
  });
});

stageOptions[0]?.classList.add("active");
bindTooltip(stageTrigger);
document.querySelectorAll(".tooltip-target").forEach((target) => {
  bindTooltip(target);
  const input = target.querySelector("input");
  input?.addEventListener("change", loadThreads);
});

newThreadButton.addEventListener("click", () => {
  const name = nextThreadName();
  setThreadName(name, null);
  closeThreadMenu();
  clearMessages();
});

deleteThreadsButton.addEventListener("click", async () => {
  const ids = [...markedThreadIds];
  if (ids.length === 0) {
    return;
  }
  const confirmed = window.confirm("是否确定删除这些聊天记录？");
  if (!confirmed) {
    return;
  }

  deleteThreadsButton.disabled = true;
  try {
    const response = await fetch(`${apiBase}/threads/delete`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({ thread_ids: ids }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "删除线程失败");
    }
    knownThreads = knownThreads.filter((thread) => !thread.id || !markedThreadIds.has(Number(thread.id)));
    markedThreadIds.clear();
    clearMessages();
    await loadThreads();
  } catch (error) {
    appendMessage("assistant", error.message);
  } finally {
    renderThreadOptions();
  }
});

customThreadButton.addEventListener("click", () => {
  const name = window.prompt("输入线程名", selectedThreadName());
  if (!name) {
    return;
  }
  setThreadName(name, null);
  closeThreadMenu();
  clearMessages();
});

document.addEventListener("click", (event) => {
  if (!stagePicker.contains(event.target)) {
    closeStageMenu();
  }
  if (!threadPicker.contains(event.target)) {
    closeThreadMenu();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) {
    return;
  }

  appendMessage("user", text);
  messageInput.value = "";
  sendButton.disabled = true;
  sendButton.textContent = "wait";

  try {
    const response = await fetch(apiBase, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      body: JSON.stringify({
        timeline_stage: selectedStage(),
        knowledge_mode: selectedMode(),
        thread_name: selectedThreadName(),
        message: text,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || `${agentName} Agent 调用失败`);
    }
    if (!knownThreads.some((thread) => thread.thread_name === data.thread_name)) {
      knownThreads.push({ id: data.thread_id, thread_name: data.thread_name });
    }
    setThreadName(data.thread_name, data.thread_id);
    appendMessage("assistant", data.answer);
  } catch (error) {
    appendMessage("assistant", error.message);
  } finally {
    sendButton.disabled = false;
    sendButton.textContent = "send";
    messageInput.focus();
  }
});

renderThreadOptions();
loadThreads();
