# ANSIO — Voice AI KOL 推荐 Agent PRD

**场景**:YC Conversational AI Hackathon(Moss 主办)
**一句话定位**:一个语音增长顾问。创始人用说话的方式描述产品,ANSIO 边聊边检索竞品投放数据,实时找出"被低估"的 KOL(Creator Alpha),并给出可执行的投放组合与 ROI 预估。
**核心叙事(给评委)**:Conversation → Multi-Hop Retrieval → Reasoning → Decision。每一句话都触发一次真实检索,每次检索 <10ms(Moss),所以语音对话全程零卡顿。

---

## 0. 可行性验证结论与必改项(2026-06-06 实测)

> 以下结论均针对真实 Moss 项目的 `kols` 索引(1000 条)live 验证,脚本见
> `agent-py/tests/moss_contract.py`(Moss 行为)与 `agent-py/tests/prd_routing_eval.py`
> (LLM 是否路由到正确工具,真 LLM gpt-4.1-mini)。实现前先各跑一遍。

**已验证可行(放心依赖):** `$eq`/`$and`/`$in` 过滤、KV 点查(`top_k=1 + $eq`)、
空结果回退、`<10ms` 延迟(实测 server 端 **0–1ms**,可写进 pitch)。`query_multi_index`
原生支持多索引检索。

**🔴 必改(不改实现一定卡住):**

1. **数值范围可下推 Moss,但「交互式调预算」要靠宽池 + Python(原则 #3 需修正)。**
   实测 Moss 对「字符串存储」的 `price_usd`/`engagement_pct` 做 `$gte/$lte/$lt` **数值比较
   完全正确**(`$gte:5000` 只返回 ≥5000)。**初次召回**预算硬过滤可下推 `$lte`,避免「语义
   top_k 全部超预算 → Python 过滤后空结果」。**但脱稿降预算(如降到 $500)不能只在缓存 top-20
   里 Python 重排**——实测同一 query 下 `$lte:500` 的 20 个结果有 **10 个不在无过滤 top-20 内**。
   推荐:`find_similar_kols` 一次性召回**宽池 `top_k=80`(不带预算 filter)**,预算过滤 + 重排放
   Python(实测 `top_k=80` 才 100% 覆盖 `$lte:500` 的 top-20,`top_k=30/50` 漏 7/3 个);这样
   降预算时「无重检索、一秒翻转」的 demo 才成立。**派生指标(Alpha/重叠/ROI)留 Python 是对的。**

2. **`$eq` 大小写 + 字符精确敏感;LLM 会传非法/自由文本的 `niche`。** 实测
   `niche="Tech"`(大写)、`platform="youtube"`(小写)都返回 **0 条**。路由评测里 LLM 给
   `find_similar_kols` 传了 `niche="developer coding workflow"`(库里根本没有)→ 永远空、
   永远走回退。**每个 `$eq` 过滤值传给 Moss 前必须:① 校验白名单 ② 规范大小写**;
   `niche` taxonomy 要和数据层统一定义一份(现有 `agent.py` 的 `KOL_PLATFORMS/KOL_NICHES`
   白名单兜底是正确做法,5 个新工具都要照搬)。

3. **`handle` 带不带 `@` 会静默返回空。** 库里存的是 `theobennett1`(无 @);
   `$eq handle="@theobennett1"` → **0 条**。PRD §3.2 `get_kol_profile` 入口必须
   `handle = handle.lstrip("@")`(或 ETL 统一带 @ 存,二选一写死)。

4. **`find_kols_who_promoted` 的「反向查 handle」会用错字段。** 路由评测里 founder 问
   "@buildwithsam 带过什么",LLM 正确路由到本工具,但把 handle 塞进了 `brand` 参数 →
   会去 filter `brand=$eq buildwithsam`(错字段)→ 0 条。**需要把品牌查询和 handle 反查
   拆成两个参数/两个工具**,filter 字段分别用 `brand` 和 `kol_handle`。

4b. **索引C(kols)的派生打分字段在现有索引里全不存在 → 复用现状会 KeyError。** 实测 live
    `kols` metadata 仅 10 个字段(`name/handle/platform/niche/tier/region/language/followers/
    engagement_pct/price_usd`),§2.1 列的 `kol_id`、`growth_3m_pct`、`audience_dev_pct`、
    `audience_founder_pct` **一个都没有**(主键用顶层 `id=kol-NNNN`,metadata 内无 `kol_id`)。
    直接复用 `gen_kols.py`/`kols.json` 会让 §3.2 `score_and_rank`(读 `growth_3m_pct`)、
    §4.1 Step 6 bundle(读 `audience_*`)**KeyError**。**二选一(详见 §3.2 注 + §2.3):
    A 重建索引补 3 字段(保「增长 × 受众构成」叙事);B 用现有 10 字段重定义
    Alpha=`engagement_pct/price_norm`、重叠=`niche/region/platform` 代理(零重建、当天可跑,
    推荐 hackathon 走 B)。**

**🟠 路由/行为实测(影响 demo 确定性):**

5. **"竞品都和谁合作?" 会扇出成 N 次调用(每个竞品一次),不是 1 次。** 两次运行稳定复现
   `find_kols_who_promoted` 连调 4 次(Cursor/Copilot/Replit/Codeium)。§4.1 Step 2 与
   右侧 `content_hits` 卡设计需支持「N 个品牌并行查 + 合并」,或在 prompt 里限定只查主竞品,
   否则 demo 节奏不可控。

6. **加约束的追问("只要 YouTube 上的")不一定触发重检索。** 两次运行都未重跑
   `find_similar_kols`(被当成确认语)。§4.2 这条要在 system prompt 里写死规则:
   「用户增删/改变 平台/赛道/受众 约束时,立刻用新 filter 重跑 `find_similar_kols`」。

**🟡 实现注意:**

7. **`load_index` 不仅为提速/过滤,更为可用性。** 未 load 时云端 fallback 路径实测直接
   `HTTP 503`,且 filter 被静默忽略(日志 `Metadata filter ignored`)。`on_enter`
   预加载四索引必须有失败重试/告警。
8. **PRD 是目标态,不是现状。** 现仓库是旧版(1 索引 + `search_kols/remember/recall`),
   5 工具/打分/门控状态机都需新建,≈ 重写 `agent.py`。
9. **门控(§3.3)需要跨轮状态载体**(槽位 + 候选缓存),建议存 `Agent` 实例字段或
   `session.userdata`,现有「单轮工具返回文本」结构撑不起「后台每轮打分」。
10. **离线无法验证 Moss**(`client.session()` 用假凭证直接 auth 失败);CI 只能跑 Python
    逻辑层的 stub 单测,工具的 live 验证必须带真凭证。`Assistant.__init__` 勿 eager 构造
    `inference.LLM`,否则连实例化都要 `LIVEKIT_API_KEY`,纯逻辑无法离线单测。

---

## 1. 系统架构

```
用户语音
   │ STT
   ▼
Voice Agent Loop (LLM + function calling)
   │                          │
   │ tool calls               │ 事件流 (SSE/WebSocket)
   ▼                          ▼
5 个工具函数              证据流 UI(右侧面板)
   │
   ├── Moss 索引 A/B/C/D(全部检索,无数据库)
   └── Python 打分模块(硬过滤 / Alpha 分 / Bundle / ROI)
   │ TTS
   ▼
语音回复
```

设计原则:

1. **全 Moss,零数据库**。语义检索、关系查找、KV 点查三种模式全部用 Moss 的 `query + filter` 组合实现。
2. **filter 管精确,向量管排序**。`$eq` metadata 过滤负责"只看投过 Cursor 的内容"这类精确约束;query 文本负责"和用户产品更相关的排前面"。
3. **数值过滤下推 Moss,派生计算留 Python**(已实测修正,见 §0.1)。Moss 对字符串
   metadata 的 `$gte/$lte/$lt/$gt` 数值比较是正确的,**预算等单字段范围过滤应下推 Moss
   `$lte`**(召回质量更高);Alpha 打分、受众重叠、ROI 外推这类「派生/多字段计算」才在
   Python 内存完成。
4. **每次 tool call 同步推一个 UI 事件**,右侧证据流与对话逐句对应。

---

## 2. Moss 索引构建

### 2.1 四个索引

#### 索引 A:`products`(竞品库)

| 部分 | 内容 |
|---|---|
| text(被向量化) | 产品介绍 + 定位描述,如 "Cursor is an AI-first code editor for professional developers, known for agentic editing and rapid iteration..." |
| metadata | `name, category, funding, stage, logo_url` |

数据量:50–100 个产品。**AI coding 赛道铺 20–30 个**(demo 主线),美妆/健身等其他赛道各铺 5–10 个防评委现场换题。

#### 索引 B:`content`(带货/合作内容库)——最重要的索引

| 部分 | 内容 |
|---|---|
| text | post 标题 + 文案 + 字幕摘要 |
| metadata | `kol_id, kol_handle, brand, platform, title, views, likes, comments, engagement_pct, date` |

要点:
- **`brand` 字段在 ETL 时由 LLM 抽取并归一化为小写**("Cursor"/"cursor.sh"/"Cursor AI" → `cursor`)。这是"哪些 KOL 投过 X"功能的基础。
- 一条内容提到多个品牌 → 每个品牌存一份文档副本(metadata 不依赖数组支持)。
- 数据量:1,500–2,000 条。Cursor 相关真实内容 ≥50 条(评委大概率是 Cursor 用户,刷到过的真实视频可信度最高),Copilot/Replit/Codeium 各 30+ 条。

#### 索引 C:`kols`(KOL 画像库)

| 部分 | 内容 |
|---|---|
| text | 画像描述文本,如 "Indie hacker focused creator, posts daily coding workflow videos, audience is 60% developers and 25% founders, casual demo style..." |
| metadata | 现状(live 实测 10 个):`handle, name, platform, niche, tier, region, language, followers, engagement_pct, price_usd`(主键用顶层 `id=kol-NNNN`,metadata 内**无 `kol_id`**)。目标态(方案 A 重建时补):再加 `growth_3m_pct, audience_dev_pct, audience_founder_pct`。完整字段类型/必填/归一化见 **§2.3**。 |

要点:
- 画像文本是 **Step 4「还有谁像他们」的检索对象**,是全 demo 最核心的一次语义检索,写 ETL 时让 LLM 把受众构成、内容风格、典型选题都写进去。
- `tier` 分桶(nano/micro/mid/macro/mega)用于替代数值范围过滤。
- 数据量:200–500 个,其中开发者向 KOL ≥80 个。

#### 索引 D:`playbook`(方法论 + 历史案例库)

| 部分 | 内容 |
|---|---|
| text | 方法论 QA("小公司为什么要参考大公司投放策略")、内容策略文档("Workflow 内容为什么比测评转化高")、历史 campaign 案例("某 AI 工具投 5 个万粉开发者 KOL,$4k 预算,trial 2,200,付费 110") |
| metadata | `doc_type(qa/strategy/case), source, campaign_reach, campaign_trials, campaign_roi` |

职责:
1. **异议处理 grounded**:Founder 质疑时,agent 的说服话术来自检索命中的文档,右侧展示出处——多数队伍这段是 LLM 现编的,这是差异点。
2. **ROI 预估有出处**:Step 8 检索相似历史案例 → 基于命中案例外推,而不是凭空报 "8–12x"(凭空 ROI 是评委最容易追问打脸的点)。

数据量:30–50 篇。

### 2.2 ETL Pipeline(赛前完成)

```
爬取(yt-dlp / Apify / 手工整理 CSV)
  → LLM 批量标注(brand 抽取归一化、niche 分类、画像文本生成)
  → 生成四份 JSONL
  → create_index 脚本写入 Moss
```

已知坑(已踩过 + 2026-06-06 实测复现,写进 checklist):
- **`load_index()` 必须先调用**,否则 filter 被静默忽略(实测日志 `Metadata filter ignored`),
  且云端 fallback 路径实测会 `HTTP 503`(不可用)。agent `on_enter` 预加载全部四个索引,
  且失败要重试/告警(见 §0.7)。
- metadata 值统一转字符串存储。**单字段数值范围过滤可直接用 Moss `$lte/$gte`(已实测正确)**;
  Alpha/重叠/ROI 等派生计算才在 Python 做。
- **`$eq` 大小写与字符精确敏感**:`platform="youtube"`、`niche="Tech"` 都返回 0 条。
  所有 filter 值在 ETL 端与查询端都要规范化(brand 统一小写;platform/niche 用白名单枚举)。
- **`handle` 统一规范**:库里存无 `@` 前缀,查询前 `lstrip("@")`(或 ETL 统一带 @,二选一)。

### 2.3 数据 Schema(JSONL)

> 每个索引一行一文档(JSONL)。Moss metadata 是 `Dict[str, str]`——**所有 metadata 值统一存字符串**(数值也存字符串,Moss 对字符串数值的 `$gte/$lte` 比较已实测正确,见 §0.1)。文档顶层 `id`/`text` 不是 metadata。归一化规则在 ETL 端与查询端必须一致(见 §0.2–0.4、§2.2)。

**枚举(与 `agent-py/src/agent.py` 的 `KOL_PLATFORMS`/`KOL_NICHES` 唯一对齐,数据层与查询层共用一份):**

- `KOL_PLATFORMS`(7,大小写精确):`YouTube`, `Instagram`, `TikTok`, `X`, `Twitch`, `Bilibili`, `LinkedIn`
- `KOL_NICHES`(20,全小写):`tech`, `gaming`, `beauty`, `fashion`, `fitness`, `food`, `finance`, `travel`, `education`, `music`, `comedy`, `lifestyle`, `parenting`, `automotive`, `business`, `crypto`, `art`, `sustainability`, `home`, `pets`
- 派生枚举 `tier`(5,由 followers 分桶):`nano`(<10k)/`micro`(<100k)/`mid`(<500k)/`macro`(<1M)/`mega`(≥1M)
- 派生枚举 `doc_type`(playbook 专用):`qa` / `strategy` / `case`

---

#### 索引 A:`products`(竞品库)

| 字段 | 位置 | 类型 | 必填 | 归一化规则 |
|---|---|---|---|---|
| `id` | 顶层 | string | 必填 | `prod-NNNN` |
| `text` | 顶层 | string(被向量化) | 必填 | 产品介绍 + 定位描述自然语言 |
| `name` | metadata | string | 必填 | 原样展示名 |
| `category` | metadata | string | 必填 | 赛道标签,小写(如 `ai-coding`、`beauty`) |
| `funding` | metadata | string | 可选 | 金额字符串(如 `400M`、`seed`) |
| `stage` | metadata | string | 可选 | 如 `seed`/`series-a`/`public` |
| `logo_url` | metadata | string | 可选 | URL |

```json
{"id":"prod-0001","text":"Cursor is an AI-first code editor for professional developers, known for agentic editing and rapid iteration over large codebases.","metadata":{"name":"Cursor","category":"ai-coding","funding":"400M","stage":"series-c","logo_url":"https://logo.cdn/cursor.png"}}
```

#### 索引 B:`content`(带货/合作内容库)

> 一条内容提到多个品牌 → 每个品牌存一份副本(metadata 不依赖数组)。

| 字段 | 位置 | 类型 | 必填 | 归一化规则 |
|---|---|---|---|---|
| `id` | 顶层 | string | 必填 | `content-NNNN` |
| `text` | 顶层 | string(被向量化) | 必填 | post 标题 + 文案 + 字幕摘要 |
| `kol_id` | metadata | string | 必填 | 关联 `kols.id`(如 `kol-0001`) |
| `kol_handle` | metadata | string | 必填 | **去 `@`** 前缀,大小写原样(如 `buildwithsam`) |
| `brand` | metadata | string | 必填 | **统一小写**,LLM 抽取归一(`Cursor`/`cursor.sh`/`Cursor AI`→`cursor`) |
| `platform` | metadata | string(enum) | 必填 | ∈ `KOL_PLATFORMS`(大小写精确) |
| `title` | metadata | string | 必填 | post 标题 |
| `views` | metadata | string(数值) | 必填 | 整数字符串 |
| `likes` | metadata | string(数值) | 可选 | 整数字符串 |
| `comments` | metadata | string(数值) | 可选 | 整数字符串 |
| `engagement_pct` | metadata | string(数值) | 必填 | 浮点字符串(如 `6.2`) |
| `date` | metadata | string | 可选 | `YYYY-MM-DD` |

```json
{"id":"content-0001","text":"How I ship 3x faster with Cursor — my full agentic workflow for refactoring legacy code.","metadata":{"kol_id":"kol-0007","kol_handle":"buildwithsam","brand":"cursor","platform":"YouTube","title":"How I ship 3x faster with Cursor","views":"890000","likes":"41000","comments":"1200","engagement_pct":"6.2","date":"2026-03-14"}}
```

#### 索引 C:`kols`(KOL 画像库)

> 当前 live `kols` 索引(1000 条)实测 metadata 字段恰为下表前 10 个,全为字符串,文档 `id` 为顶层 `kol-NNNN`(metadata 内**无** `kol_id`)。`growth_3m_pct`/`audience_dev_pct`/`audience_founder_pct` 是目标态新增字段(供 Alpha 打分与受众重叠用,§0.4b),ETL 重建(方案 A)时补齐。

| 字段 | 位置 | 类型 | 必填 | 归一化规则 |
|---|---|---|---|---|
| `id` | 顶层 | string | 必填 | `kol-NNNN`(= content.kol_id 外键) |
| `text` | 顶层 | string(被向量化) | 必填 | 画像描述(受众构成/内容风格/典型选题写进去,§2.1) |
| `name` | metadata | string | 必填 | 展示名 |
| `handle` | metadata | string | 必填 | **去 `@`** 前缀(查询前 `lstrip("@")`,§0.3) |
| `platform` | metadata | string(enum) | 必填 | ∈ `KOL_PLATFORMS`(大小写精确) |
| `niche` | metadata | string(enum) | 必填 | ∈ `KOL_NICHES`(全小写) |
| `tier` | metadata | string(enum) | 必填 | ∈ `{nano,micro,mid,macro,mega}`,按 followers 分桶 |
| `region` | metadata | string | 必填 | 如 `United States` |
| `language` | metadata | string | 必填 | 如 `English` |
| `followers` | metadata | string(数值) | 必填 | 整数字符串 |
| `engagement_pct` | metadata | string(数值) | 必填 | 浮点字符串 |
| `price_usd` | metadata | string(数值) | 必填 | 整数字符串(预算 `$lte` 下推用) |
| `growth_3m_pct` | metadata | string(数值) | 可选(目标态) | 浮点字符串,Alpha 分子(方案 A) |
| `audience_dev_pct` | metadata | string(数值) | 可选(目标态) | 浮点字符串,受众重叠用(方案 A) |
| `audience_founder_pct` | metadata | string(数值) | 可选(目标态) | 浮点字符串,受众重叠用(方案 A) |

```json
{"id":"kol-0001","text":"Avery Cruz (@averycruz) is a top education creator on Instagram with 1.3M followers, based in United States, posting in English. Their content covers study tips, edutainment, tutorials, language learning, mostly as short-form videos. Average engagement rate 1.9%. Audience tier: mega.","metadata":{"name":"Avery Cruz","handle":"averycruz","platform":"Instagram","niche":"education","tier":"mega","region":"United States","language":"English","followers":"1285435","engagement_pct":"1.9","price_usd":"19574"}}
```

#### 索引 D:`playbook`(方法论 + 历史案例库)

| 字段 | 位置 | 类型 | 必填 | 归一化规则 |
|---|---|---|---|---|
| `id` | 顶层 | string | 必填 | `pb-NNNN` |
| `text` | 顶层 | string(被向量化) | 必填 | 方法论 QA / 内容策略 / 历史 campaign 案例正文 |
| `doc_type` | metadata | string(enum) | 必填 | ∈ `{qa,strategy,case}`(`$eq` filter 用) |
| `source` | metadata | string | 必填 | 出处(右侧卡片展示) |
| `campaign_reach` | metadata | string(数值) | 可选(`case` 用) | 整数字符串 |
| `campaign_trials` | metadata | string(数值) | 可选(`case` 用) | 整数字符串 |
| `campaign_roi` | metadata | string(数值) | 可选(`case` 用) | 浮点字符串(如 `8.5`) |

```json
{"id":"pb-0001","text":"某 AI 工具投放 5 个万粉级开发者 KOL,$4,000 预算,获得 2,200 次 trial、110 个付费转化,验证了 micro-influencer workflow 内容的转化效率。","metadata":{"doc_type":"case","source":"内部 campaign 复盘 2025Q4","campaign_reach":"520000","campaign_trials":"2200","campaign_roi":"8.5"}}
```

---

## 3. Function Call 设计

### 3.1 工具总览(5 个)

| 工具 | 索引 | 模式 | 服务的对话环节 |
|---|---|---|---|
| `find_competitors` | A | 纯语义 | Step 1 竞品发现 |
| `find_kols_who_promoted` | B | `$eq` filter + Python 聚合 | Step 2 谁投过 / Step 7 内容角度 |
| `get_kol_profile` | C | `$eq` filter 点查(KV 模式) | Step 3 拉画像 |
| `find_similar_kols` | C | 语义 + 可选分类 filter | Step 4 扩相似达人 |
| `search_playbook` | D | 纯语义 | 异议处理 / Step 8 ROI 案例 |

打分(Alpha)、组合(Bundle)、ROI 外推是**纯 Python 函数**,不是检索工具;它们消费上述工具的返回值。

### 3.2 工具定义与实现要点

```python
# 工具 1:竞品发现 —— 纯语义
async def find_competitors(product_desc: str, top_k: int = 5):
    r = await client.query("products", product_desc, QueryOptions(top_k=top_k))
    if not r.docs:                                   # 空结果 → 返回空,LLM 口头兜底(§5)
        return []
    return [{"name": d.metadata["name"], "category": d.metadata["category"],
             "funding": d.metadata.get("funding", ""),   # funding 可选,避免 KeyError
             "score": d.score} for d in r.docs]

# 工具 2:谁投过 / 反查 handle —— filter 精确 + 向量排序 + 聚合
# 注意(实测):正向"谁投过 X"按 brand 过滤;反向"@handle 带过什么"按 kol_handle 过滤,
# 必须用不同字段,否则反查会 filter brand=$eq <handle> 拿到 0 条(见 §0.4)。
async def find_kols_who_promoted(brand: str = "", kol_handle: str = "",
                                 product_desc: str = ""):
    if kol_handle:                                  # 反查模式
        field, val = "kol_handle", kol_handle.lstrip("@")
    else:                                           # 正向模式
        field, val = "brand", brand.lower()         # brand 在 ETL 已归一化小写
    r = await client.query(
        "content",
        product_desc or f"{brand or kol_handle} review",  # 向量负责"更相关的排前面"
        QueryOptions(
            top_k=50,                               # post 粒度,放大后聚合
            filter={"field": field, "condition": {"$eq": val}},
        ))
    if not r.docs:                                  # 品牌/handle 无命中 → 空,LLM 口头兜底(§5)
        return []
    kols = {}
    for d in r.docs:
        m = d.metadata
        k = kols.setdefault(m["kol_id"], {"handle": m["kol_handle"],
                                          "posts": [], "total_views": 0})
        k["posts"].append({"title": m["title"], "views": int(m["views"]),
                           "engagement_pct": float(m["engagement_pct"])})
        k["total_views"] += int(m["views"])
    return sorted(kols.values(), key=lambda x: -x["total_views"])[:10]

# 工具 3:画像点查 —— Moss 当 KV 用
async def get_kol_profile(handle: str):
    handle = handle.lstrip("@")                      # 实测:带 @ 会 0 命中(§0.3)
    r = await client.query("kols", handle, QueryOptions(
        top_k=1, filter={"field": "handle", "condition": {"$eq": handle}}))
    if not r.docs:                                   # LLM 可能传错/旧 handle
        return None
    return r.docs[0].metadata | {"profile_text": r.docs[0].text}

# 工具 4:扩相似达人 —— 核心语义检索
async def find_similar_kols(profile_text: str, niche: str = None,
                            platform: str = None, top_k: int = 20):
    # 实测:LLM 会传非法 niche(如 "developer coding workflow")→ $eq 永远空(§0.2)。
    # 必须白名单校验 + 规范大小写,非法值直接置空(退化为纯语义)。
    niche = niche if niche in KOL_NICHES else None
    platform = platform if platform in KOL_PLATFORMS else None
    r = await client.query("kols", profile_text,
                           QueryOptions(top_k=top_k,
                                        filter=_build_filter(platform, niche)))
    if not r.docs and (niche or platform):        # 空结果回退:去掉 filter 重试
        r = await client.query("kols", profile_text, QueryOptions(top_k=top_k))
    return [d.metadata | {"sim": d.score} for d in r.docs]

# 工具 5:方法论/案例检索
async def search_playbook(question: str, doc_type: str = None):
    f = {"field": "doc_type", "condition": {"$eq": doc_type}} if doc_type else None
    r = await client.query("playbook", question, QueryOptions(top_k=3, filter=f))
    return [{"text": d.text, "source": d.metadata["source"],
             "meta": d.metadata} for d in r.docs]
```

```python
# 打分模块(纯 Python,非工具)
# 预算:find_similar_kols 用宽池 top_k=80 召回(不下推预算 filter),预算过滤放这里,
# 这样脱稿调预算无需重检索、可秒级 FLIP 重排(实测依据见 §0.1)。
def score_and_rank(candidates, slots, weights=None):
    w = weights or {"match": 0.4, "perf": 0.3, "alpha": 0.3}
    pool = [c for c in candidates
            if int(c["price_usd"]) <= slots["budget_per_video"]]   # 预算硬过滤
    for c in pool:
        # Alpha = 被低估度。方案 A(重建索引,§0.4b):用 growth_3m_pct(增长快但还没贵);
        #        方案 B(现有 10 字段,推荐):用 engagement_pct(高互动 / 低报价 = 被低估)。
        numer = float(c.get("growth_3m_pct") or c["engagement_pct"])
        c["alpha"] = numer / max(price_norm(c), 1e-6)
    # 各项 min-max 归一化后加权
    return sorted(pool, key=lambda c: total(c, w), reverse=True)[:5]
```

### 3.3 推荐门控(什么时候开口推)

后台每轮都在检索和打分,但**对用户开口推荐**需满足任一:
1. **槽位收敛**:必填槽位 `{产品品类, 目标受众, 平台, 预算, 目标(转化/曝光)}` 抽取齐全;
2. **候选稳定**:连续两轮 top-5 重合 ≥4/5;
3. **用户催促**:"直接推荐吧" → 立即推;
4. **轮次上限**:第 5 轮强制给初步结果。

第 2–3 轮渐进透出:"我这边已经看到几个挺匹配的候选了,再确认下预算范围。"

### 3.4 状态与记忆(边界说清,避免和 5 工具混淆)

| 类型 | 存哪 | 生命周期 | 干什么 |
|---|---|---|---|
| **会话内槽位 + 候选缓存** | `Agent` 实例字段 / `session.userdata`(**非 Moss**,§0.9) | 单次通话,挂断即清 | 槽位收敛判定(§3.3)、宽池候选缓存(供调预算/权重 Python 重排,§0.1) |
| **跨会话记忆(可选,非主线)** | 现有 `memory` Moss 索引 + `remember_fact/recall_facts`(per-user `$eq user_id` 过滤,已 live 验证) | 跨通话持久 | "记住我的品牌/上次的 brief",下次开口直接复用 |

**结论**:demo 主线只需**会话内状态**,5 个检索工具都是无状态的;`memory` 索引与 `remember_fact/recall_facts` **不属于 5 工具**,是可选的第 6/7 工具(加分项,现 `agent.py` 已实现,可保留)。**若不做跨会话记忆,可以不 load `memory` 索引**——但 `on_enter` 仍需 load `products/content/kols/playbook` 四索引(§0.7)。

---

## 4. 对话 → Tool Call 映射示例

### 4.1 主线剧本(8 步,对应 demo 脚本)

| # | 用户/Agent 说什么 | Agent 调用 | 参数(示例) | 右侧 UI 事件 |
|---|---|---|---|---|
| 1 | F:"我们是做 AI Coding Tool 的,增长卡住了" | `find_competitors` | `product_desc="AI coding tool / code editor for developers and indie hackers"` | `competitor_landscape` 卡:Cursor / Copilot / Replit / Codeium + funding 徽章 + `7ms` |
| 2 | F:"他们都在和谁合作?" | `find_kols_who_promoted` ×N(实测 LLM 每个竞品各调一次,§0.5) | `brand="cursor"`(及 `copilot`/`replit`/`codeium`),`product_desc="AI coding tool for indie developers"` | `content_hits` 卡:**N 个品牌合并**的命中 post 列表 + 聚合出的合作 KOL 名单 + `9ms` |
| 2b | F:"Cursor 体量太大,对标有意义吗?"(异议) | `search_playbook` | `question="小公司为什么要参考头部公司的投放策略", doc_type="qa"` | `playbook_hit` 卡:命中文档摘录 + 出处 |
| 3 | A:"我先拆解这几位的画像" | `get_kol_profile` ×3 | `handle="buildwithsam"`(去 @,§0.3)等 | `kol_profile` 卡:受众构成 / 内容风格 / 历史表现 |
| 4 | F:"还有类似的人吗?" | `find_similar_kols` | `profile_text=<Step3 画像文本拼接>, niche="tech"`(合法枚举,§2.3;`dev_tools` 不存在) | `similar_creators` 卡:20 个候选 + 相似度 |
| 5 | A:"看看谁被低估了" | (无检索)`score_and_rank` | `candidates=<Step4 返回>, slots={budget:1000,...}` | `alpha_ranking` 卡:Alpha 分拆解(match/perf/alpha 三列) |
| 6 | A:"不建议只投一个人" | (无检索)Python 算受众重叠 | 方案 A:`audience_dev_pct/founder_pct`;方案 B(推荐):`niche/region/platform` 代理重叠(§0.4b) | `bundle` 卡:A+B 组合,overlap / 覆盖 / 预算分配 |
| 7 | A:"内容别做测评,做 Workflow" | `find_kols_who_promoted`(复用) + `search_playbook` | `brand="cursor"` 取高转化内容样本;`question="developer tool 什么内容形式转化最高"` | `content_strategy` 卡:Top 内容角度 + 例子 hook |
| 8 | F:"值得投吗?" | `search_playbook` | `question="AI developer tool micro influencer campaign 历史 ROI", doc_type="case"` | `roi_forecast` 卡:命中案例 + 外推 Reach/Trials/ROI,标注"基于 N 个相似案例" |

### 4.2 脱稿场景(评委自由提问时的鲁棒性)

| 用户说 | Agent 行为 |
|---|---|
| "只要 YouTube 上的" | `find_similar_kols(profile_text=..., platform="YouTube")` —— 复合 filter `$and: [platform, niche]` |
| "预算改成每条 $500" | 无检索(前提:`find_similar_kols` 已用宽池 `top_k=80` 召回、预算过滤在 Python 侧,§0.1)。更新 slot → Python 对宽池做 `price_usd<=budget` 过滤 + `score_and_rank` 重排 → UI 推送 `alpha_ranking`(FLIP 动画)。⚠️ 若初次召回已下推了更高预算 `$lte`,降到其下需重跑 `find_similar_kols` |
| "我更看重性价比" | 无检索。`weights={"alpha":0.6,...}` 重排 → 这是 demo 高潮:排名一秒翻转 |
| "我们其实是做 K-beauty 的"(换赛道) | 全链路同样工作:`find_competitors("sensitive skin repair serum K-beauty")` → 命中美妆竞品(所以 A 库要铺多赛道) |
| "@buildwithsam 带过什么?" | `find_kols_who_promoted(kol_handle="buildwithsam")` 反向用:索引 B 上 filter `kol_handle=$eq buildwithsam`(**去 @**,§0.3/§0.4),query 留空品类词 |
| "为什么是这两个人?" | `search_playbook(doc_type="strategy")` + 复述 Alpha 拆解,不现编 |

### 4.3 一次完整 tool call 的 UI 事件格式

```json
{ "type": "content_hits",
  "step": 2,
  "latency_ms": 9,
  "index": "content",
  "items": [{"kol": "@buildwithsam", "title": "How I ship 3x faster with Cursor",
             "views": 890000, "engagement_pct": 6.2}],
  "insight": "Cursor 的高转化合作集中在 workflow 实录类内容,而非产品测评" }
```

`latency_ms` 取真实计时,渲染为卡片右上角徽章(给赞助商看的)。`insight` 是每张卡唯一的 LLM 自由文本字段,一句话。

---

## 5. 非功能要求

| 项 | 目标 |
|---|---|
| 单次 Moss 查询 | <10ms(真实测量,写进 pitch) |
| 用户说完 → 开始回答 | <800ms(检索不是瓶颈,瓶颈在 LLM 首 token + TTS) |
| 调权重重排 | <1s,带 FLIP 动画 |
| 空结果 | filter 命中 0 → 自动去 filter 重试;仍为 0 → agent 口头兜底("这个赛道我库里数据少,先按相邻赛道给你参考") |
| 网络故障 | 启动(建 session + `load_index`)仍需联网——**上台前用稳定网络预热加载好四索引**。`load_index` 成功后,只读过滤查询走内存,**现场 Wi-Fi 断了也能继续检索**;但写入/记忆(按 user_id 写)仍需云端。注:未 load 的查询路径实测返回 HTTP 503 且忽略 filter,故必须先 load 成功(§0.7)。 |

---

## 6. 赛前 Checklist

按依赖排序,**ETL 是关键路径,最先做**:

- [ ] **🔴 P0 ETL(决定生死)**:爬 Cursor/Copilot/Replit/Codeium 相关内容 ≥150 条(真实),
      LLM 标注并**归一化** brand(小写)/niche(白名单枚举)/handle(去 @)。
      `content` 索引是差异化核心,空了整个 demo 叙事不成立 → 准备**合成数据兜底**方案。
- [ ] 四个索引(`products/content/kols/playbook`)建好;跑 `tests/moss_contract.py`
      验证 `$eq`/`$and`/数值范围/handle 规范/load_index 都符合预期(目前只有 `kols` 存在)
- [ ] 5 个工具 + `_build_filter` + 白名单校验 + 空结果回退接入 agent.py;
      `find_kols_who_promoted` 拆 brand / kol_handle 两参数(§0.4)
- [ ] 跑 `tests/prd_routing_eval.py` 确认 LLM 路由正确;重点修「加约束追问不重检索」(§0.6)、
      「竞品合作查询扇出 N 次」的 UI/节奏处理(§0.5)
- [ ] 打分模块单测(stub MossClient,离线):预算硬过滤、权重重排;Assistant 勿 eager 建 LLM(§0.10)
- [ ] 槽位抽取 prompt + 门控逻辑 + 跨轮状态载体(§0.9)
- [ ] 证据流 UI:7 种卡片组件 + SSE 事件流 + FLIP 动画
- [ ] 端到端延迟实测,数字写进 pitch deck(Moss server 端实测 0–1ms,可直接引用)
- [ ] 脱稿演练:换赛道、改预算、改权重、反向查询各过一遍
