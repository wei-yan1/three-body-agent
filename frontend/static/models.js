const $ = (selector) => document.querySelector(selector);
const nf = new Intl.NumberFormat("zh-CN");
const stageNames = {
  parse: "解析文档", chunk: "切分文本", embedding_and_index: "向量化与建索引",
  "a2a:query-planner": "问题规划", "a2a:deep-question-planner": "深度问题规划",
  "a2a:context-curator": "上下文整理", "a2a:relationship-state": "关系状态读取",
  "a2a:relationship-update": "关系状态更新", "a2a:timeline-guard": "时间线检查",
  "a2a:evidence-analysis": "证据分析", "a2a:role-cognition": "角色认知",
  "a2a:character-reasoning": "角色推理", "a2a:nuwa-profiler": "女娲角色画像",
  "a2a:memory-decision": "记忆决策", "a2a:consistency-guard": "一致性审查",
  "a2a:character-period-analysis": "时期分析", "a2a:web-search": "联网检索",
  answer_generation: "回答生成", query_planning: "问题规划", retrieval: "小说检索",
  context_curator: "上下文整理", reasoning: "角色推理", consistency_guard: "一致性审查",
  persistence: "结果持久化",
};
function token() { return localStorage.getItem("access_token"); }
function number(value) { return nf.format(Number(value || 0)); }
function ms(value) { if (value == null) return "-"; const n = Number(value); return n >= 1000 ? `${(n / 1000).toFixed(2)} s` : `${Math.round(n)} ms`; }
function cost(value) { return value == null ? "未配置" : Number(value).toFixed(4); }
function esc(value) { return String(value ?? "-").replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char])); }
function stageName(value) { return stageNames[value] || value || "未命名阶段"; }
function statusText(value) { return ({ completed: "已完成", failed: "失败", running: "进行中", degraded: "降级" }[value] || value || "-"); }

async function request() {
  const currentToken = token();
  if (!currentToken) { location.href = "/"; return new Promise(() => {}); }
  const response = await fetch("/api/v1/models/dashboard?days=14", { headers: { Authorization: `Bearer ${currentToken}` } });
  const data = await response.json();
  if (response.status === 401) { localStorage.removeItem("access_token"); location.href = "/"; return new Promise(() => {}); }
  if (!response.ok) throw Error(data.detail || "无法读取调用数据");
  return data;
}

function render(data) {
  const totals = data.totals || {};
  const runs = data.pipeline_runs || [];
  const stages = data.pipeline_stage_summary || [];
  $("#period").textContent = `最近 ${data.days} 天`;
  $("#pipeline-window").textContent = `最近 ${data.days} 天`;
  $("#totals").innerHTML = [
    ["模型调用", number(totals.calls), `${number(totals.errors)} 次失败`],
    ["总 token", number(totals.total_tokens), `输入 ${number(totals.input_tokens)} / 输出 ${number(totals.output_tokens)}`],
    ["总费用", cost(totals.total_cost), "按 MODEL_PRICING_JSON 计算"],
    ["P95 延迟", ms(totals.p95_latency_ms), `平均 ${ms(totals.avg_latency_ms)}`],
    ["缓存命中", `${(Number(totals.cache_hit_rate || 0) * 100).toFixed(1)}%`, `命中 ${number(totals.cache_hits)} 次`],
  ].map((item) => `<article class="metric"><small>${item[0]}</small><strong>${item[1]}</strong><span>${item[2]}</span></article>`).join("");

  const daily = data.daily || [];
  const maxCalls = Math.max(1, ...daily.map((item) => Number(item.calls || 0)));
  $("#daily").innerHTML = daily.length ? daily.map((item) => `<div class="day"><b>${String(item.usage_date).slice(5)}</b><span class="bar"><i style="width:${Math.min(100, Number(item.calls || 0) / maxCalls * 100)}%"></i></span><em>${number(item.calls)} 次 · ${number(item.total_tokens)} tokens</em></div>`).join("") : `<p class="empty">暂无调用记录</p>`;
  const cache = data.embedding_cache || {};
  $("#cache").innerHTML = `<strong>${number(cache.entries)}</strong><span>条向量缓存</span><small>缓存维度记录 ${number(cache.dimensions)}</small><p>导入和检索会按模型版本与文本指纹复用向量，重复运行不会重复请求 embedding 服务。</p>`;

  const agents = data.agents || [];
  $("#agents").innerHTML = agents.length ? agents.map((item) => `<div class="agent-item"><div><strong>${esc(item.agent_id || "未命名 Agent")}</strong><small>${number(item.calls)} 次调用 · ${number(item.total_tokens)} tokens · ${number(item.errors)} 次失败</small></div><span>${ms(item.avg_latency_ms)} 平均</span></div>`).join("") : `<p class="empty">暂无 Agent 调用记录</p>`;

  const completedRuns = runs.filter((run) => run.status === "completed" || run.status === "degraded");
  const avgRun = completedRuns.length ? completedRuns.reduce((sum, run) => sum + Number(run.duration_ms || 0), 0) / completedRuns.length : 0;
  const slowestRun = runs.reduce((slowest, run) => Number(run.duration_ms || 0) > Number(slowest?.duration_ms || 0) ? run : slowest, null);
  $("#run-metrics").innerHTML = [
    ["最近运行", number(runs.length), "条流程记录"], ["平均耗时", ms(avgRun), `${number(completedRuns.length)} 条已结束`],
    ["最长运行", ms(slowestRun?.duration_ms), slowestRun ? stageName(slowestRun.run_type) : "暂无"],
    ["流程失败", number(runs.filter((run) => run.status === "failed").length), "需要关注"],
  ].map((item) => `<div class="run-metric"><small>${item[0]}</small><strong>${item[1]}</strong><span>${item[2]}</span></div>`).join("");

  const maxStage = Math.max(1, ...stages.map((stage) => Number(stage.avg_duration_ms || 0)));
  $("#pipeline-stages").innerHTML = stages.length ? stages.map((stage) => `<div class="stage-item"><div class="stage-label"><strong>${esc(stageName(stage.step_name))}</strong><small>${esc(stage.step_name)} · ${number(stage.calls)} 次 · P95 ${ms(stage.p95_duration_ms)}</small></div><div class="stage-track"><i style="width:${Math.max(2, Number(stage.avg_duration_ms || 0) / maxStage * 100)}%"></i></div><span>${ms(stage.avg_duration_ms)}</span></div>`).join("") : `<p class="empty">暂无流程阶段记录。完成一次导入或对话后，这里会显示耗时拆解。</p>`;
  $("#pipeline-runs").innerHTML = runs.length ? runs.map((run) => `<div class="run-item"><div><strong>${esc(stageName(run.run_type))}</strong><small>${new Date(run.started_at).toLocaleString()} · ${number(run.model_call_count)} 次模型调用 · ${number(run.total_tokens)} tokens</small></div><span class="run-status ${esc(run.status)}">${statusText(run.status)}<b>${ms(run.duration_ms)}</b></span></div>`).join("") : `<p class="empty">暂无流程运行记录</p>`;

  $("#recent").innerHTML = (data.recent || []).length ? data.recent.map((item) => `<tr><td>${new Date(item.started_at).toLocaleString()}</td><td>${esc(item.agent_id || "-")}<small class="operation">${esc(item.operation || "-")}</small></td><td>${esc(item.model || "-")}</td><td><span class="status ${esc(item.status)}">${item.cache_hit ? "缓存命中" : statusText(item.status)}</span></td><td>${number(item.total_tokens)}</td><td>${ms(item.duration_ms)}</td><td>${cost(item.cost)}</td></tr>`).join("") : `<tr><td colspan="7" class="empty">暂无调用记录</td></tr>`;
  $("#updated").textContent = `更新于 ${new Date().toLocaleTimeString()}`;
}
async function load() { try { $("#error").classList.add("hidden"); render(await request()); } catch (error) { $("#error").textContent = error.message; $("#error").classList.remove("hidden"); } }
$("#refresh").onclick = load;
$("#logout")?.addEventListener("click", () => { localStorage.removeItem("access_token"); localStorage.removeItem("current_user"); location.href = "/"; });
load();
setInterval(load, 30000);
