// ===================== 状态 =====================
// 每条消息：{ role, content, reasoning?, query?, chunks? }，后两者只存在于 assistant 消息上
let messages = [];
let busy = false;

const $ = (id) => document.getElementById(id);
const el = (tag, cls, html) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
};
const escapeHtml = (s) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

// CommonMark 规定 ** 紧挨中文标点时不算加粗（例如 **溢出效应（spillover）**），这里先手动转成 <strong>；
// 代码块和行内代码里的 ** 保持原样（split 的捕获组让奇数下标正好是代码部分）
function fixCjkBold(text) {
  return text
    .split(/(```[\s\S]*?```|`[^`\n]*`)/)
    .map((part, i) => (i % 2 ? part : part.replace(/\*\*([^*\n]+?)\*\*/g, "<strong>$1</strong>")))
    .join("");
}

// Markdown 渲染；把 [1]、[2,3] 这样的引用变成可点击的小标签
function renderMarkdown(text) {
  let html = window.marked && window.DOMPurify
    ? DOMPurify.sanitize(marked.parse(fixCjkBold(text)))
    : escapeHtml(text).replace(/\n/g, "<br>");
  return html.replace(/\[(\d+(?:\s*[,，]\s*\d+)*)\]/g, (_, nums) =>
    nums.split(/[,，]/).map((n) => `<span class="cite" data-n="${n.trim()}">${n.trim()}</span>`).join("")
  );
}

// ===================== 索引状态 / 设置 =====================
async function loadStatus() {
  const s = await fetch("/api/status").then((r) => r.json());
  $("models").textContent = `LLM: ${s.llm_model} · Embedding: ${s.embedding_model}`;

  const method = $("method");
  if (!method.options.length) {
    s.chunk_methods.forEach((m) => method.add(new Option(m, m)));
    const cur = s.index || {};
    method.value = cur.chunk_method || s.defaults.method;
    $("chunkSize").value = cur.chunk_size || s.defaults.chunk_size;
    $("chunkOverlap").value = cur.chunk_overlap ?? s.defaults.chunk_overlap;
    $("topK").value = s.defaults.top_k;
  }
  $("filesHint").textContent = `${s.papers_dir} 下共 ${s.files.length} 个文件`;
  // 只读模式（公网部署）：隐藏建索引相关的设置
  $("indexGroup").classList.toggle("hidden", !!s.read_only);

  const box = $("indexStatus");
  if (s.index && s.index.embedding_model !== s.embedding_model) {
    // 换了 embedding 模型却没重建索引：向量维度/空间不一致，检索会出错
    box.className = "index-status warn";
    box.textContent = `索引由 ${s.index.embedding_model} 构建，与当前模型 ${s.embedding_model} 不一致，请重建索引`;
  } else if (s.index && s.index.num_chunks_in_db > 0) {
    const i = s.index;
    box.className = "index-status";
    box.innerHTML = `知识库：<b>${i.num_docs}</b> 篇论文 · <b>${i.num_chunks_in_db}</b> 个 chunk · 切块 <b>${i.chunk_method}</b>(${i.chunk_size}/${i.chunk_overlap}) · ${i.built_at}`;
    box.title = i.documents.map((d) => `${d.source}  —  ${d.title}`).join("\n");
  } else {
    box.className = "index-status warn";
    box.textContent = `尚未建立索引：把论文放进 ${s.papers_dir}，然后在「设置」里点击「重建索引」`;
  }
}

async function rebuildIndex() {
  const btn = $("rebuild");
  btn.disabled = true;
  btn.textContent = "构建中…（需要调用 embedding API）";
  try {
    const r = await fetch("/api/index", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        method: $("method").value,
        chunk_size: +$("chunkSize").value,
        chunk_overlap: +$("chunkOverlap").value,
      }),
    });
    const data = await r.json();
    if (!r.ok) throw new Error(data.detail || r.statusText);
    alert(`索引完成：${data.num_docs} 篇论文，${data.num_chunks} 个 chunk，平均 ${data.avg_chunk_chars} 字符，用时 ${data.seconds}s`);
  } catch (e) {
    alert("建索引失败：" + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = "重建索引";
    loadStatus();
  }
}

// ===================== 右侧：检索结果 =====================
function renderChunks(msg) {
  const box = $("chunks");
  box.innerHTML = "";
  const chunks = msg?.chunks;
  $("resultCount").textContent = chunks ? `Top ${chunks.length}` : "";

  const q = $("queryUsed");
  if (msg?.query) {
    q.classList.remove("hidden");
    q.textContent = `检索查询：${msg.query}`;
  } else q.classList.add("hidden");

  if (!chunks) {
    box.append(el("div", "empty", "检索中…"));
    return;
  }
  if (!chunks.length) {
    box.append(el("div", "empty", "没有检索到内容，请先建立索引。"));
    return;
  }
  for (const c of chunks) {
    const pages = c.page_start === c.page_end ? `p.${c.page_start}` : `p.${c.page_start}-${c.page_end}`;
    const pct = Math.max(0, Math.min(1, c.score)) * 100;
    const card = el("div", "chunk");
    card.id = `chunk-${c.rank}`;
    card.innerHTML = `
      <div class="chunk-head">
        <div class="rank">${c.rank}</div>
        <div class="score" title="余弦相似度">
          <div class="bar"><span style="width:${pct}%"></span></div>
          <span class="num">${c.score.toFixed(4)}</span>
        </div>
      </div>
      <div class="source"><b>${escapeHtml(c.source)}</b> · ${pages} · chunk #${c.chunk_id.split("::").pop()}<br>${escapeHtml(c.title)}</div>
      <div class="chunk-text">${escapeHtml(c.text)}</div>
      <div class="toggle">展开全文</div>`;
    card.querySelector(".toggle").onclick = (e) => {
      card.classList.toggle("expanded");
      e.target.textContent = card.classList.contains("expanded") ? "收起" : "展开全文";
    };
    box.append(card);
  }
}

function focusChunk(n) {
  const card = $(`chunk-${n}`);
  if (!card) return;
  card.scrollIntoView({ behavior: "smooth", block: "center" });
  card.classList.add("flash");
  setTimeout(() => card.classList.remove("flash"), 1200);
}

// ===================== 左侧：对话 =====================
function renderMessage(msg, idx) {
  if (msg.role === "user") return el("div", "msg user", escapeHtml(msg.content));

  const div = el("div", "msg assistant");
  div.dataset.idx = idx;
  if (msg.reasoning) {
    const d = el("details", "reasoning");
    d.innerHTML = `<summary>思考过程</summary><div>${escapeHtml(msg.reasoning)}</div>`;
    div.append(d);
  }
  const body = el("div", "body" + (msg.streaming ? " typing" : ""));
  body.innerHTML = msg.error
    ? `<span class="error">⚠ ${escapeHtml(msg.error)}</span>`
    : msg.content ? renderMarkdown(msg.content) : msg.streaming ? (msg.chunks ? "生成中" : "检索中") : "";
  div.append(body);

  // 点击某条回答：右侧切换为这条回答对应的检索结果；点击引用标签：定位到对应片段
  div.onclick = (e) => {
    selectAssistant(idx);
    const cite = e.target.closest(".cite");
    if (cite) focusChunk(cite.dataset.n);
  };
  return div;
}

function renderMessages() {
  const box = $("messages");
  $("emptyChat").classList.toggle("hidden", messages.length > 0);
  box.querySelectorAll(".msg").forEach((m) => m.remove());
  messages.forEach((m, i) => box.append(renderMessage(m, i)));
  box.scrollTop = box.scrollHeight;
}

// 流式更新时只重绘最后一条消息，避免整页闪烁
function updateLast() {
  const box = $("messages");
  const idx = messages.length - 1;
  const old = box.querySelector(`.msg.assistant[data-idx="${idx}"]`);
  const node = renderMessage(messages[idx], idx);
  if (old) {
    const open = old.querySelector("details")?.open;
    if (open && node.querySelector("details")) node.querySelector("details").open = true;
    node.classList.toggle("active", old.classList.contains("active"));
    old.replaceWith(node);
  } else box.append(node);
  const nearBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 120;
  if (nearBottom) box.scrollTop = box.scrollHeight;
}

function selectAssistant(idx) {
  document.querySelectorAll(".msg.assistant").forEach((m) => m.classList.toggle("active", +m.dataset.idx === idx));
  renderChunks(messages[idx]);
}

// 解析 Server-Sent Events：每个事件形如 "event: xxx\ndata: {...}\n\n"
async function* readSSE(response) {
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let pos;
    while ((pos = buf.indexOf("\n\n")) >= 0) {
      const raw = buf.slice(0, pos);
      buf = buf.slice(pos + 2);
      let event = "message", data = "";
      for (const line of raw.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      yield { event, data: data ? JSON.parse(data) : null };
    }
  }
}

async function send(text) {
  text = text.trim();
  if (!text || busy) return;
  busy = true;
  $("send").disabled = true;
  $("input").value = "";

  messages.push({ role: "user", content: text });
  const reply = { role: "assistant", content: "", reasoning: "", streaming: true };
  messages.push(reply);
  renderMessages();
  const idx = messages.length - 1;
  selectAssistant(idx);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        // 只把纯文本对话发给后端，历史检索结果不需要回传
        messages: messages.slice(0, -1).map(({ role, content }) => ({ role, content })),
        top_k: +$("topK").value,
        rewrite: $("rewrite").checked,
        thinking: $("thinking").checked,
      }),
    });
    if (!res.ok) throw new Error((await res.json()).detail || res.statusText);

    for await (const { event, data } of readSSE(res)) {
      if (event === "query") {
        reply.query = data;
      } else if (event === "chunks") {
        reply.chunks = data;
        renderChunks(reply);
      } else if (event === "reasoning") {
        reply.reasoning += data;
      } else if (event === "content") {
        reply.content += data;
      } else if (event === "error") {
        reply.error = data.message;
      }
      updateLast();
    }
  } catch (e) {
    reply.error = e.message;
  } finally {
    reply.streaming = false;
    if (!reply.chunks) reply.chunks = [];
    updateLast();
    selectAssistant(idx);
    busy = false;
    $("send").disabled = false;
    $("input").focus();
  }
}

// ===================== 事件绑定 =====================
$("composer").onsubmit = (e) => {
  e.preventDefault();
  send($("input").value);
};
$("input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) {
    e.preventDefault();
    send($("input").value);
  }
});
document.querySelectorAll(".example").forEach((li) => (li.onclick = () => send(li.textContent)));
$("newChat").onclick = () => {
  if (busy) return;
  messages = [];
  renderMessages();
  $("chunks").innerHTML = '<div class="empty">提问后，这里会展示从知识库中召回的片段、排名和相关性分数。</div>';
  $("resultCount").textContent = "";
  $("queryUsed").classList.add("hidden");
};
$("toggleSettings").onclick = () => $("settings").classList.toggle("hidden");
$("rebuild").onclick = rebuildIndex;

loadStatus().catch((e) => ($("indexStatus").textContent = "无法连接后端：" + e.message));
