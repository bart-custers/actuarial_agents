# AZD Multi-Agent Actuarial Risk Prediction POC — Plan
### Non-motor (Home/Household), Claude-based, delivered as a Claude Code Plugin

## Context

AZD wants a POC multi-agent AI system for actuarial risk prediction on non-motor lines
(Home/Household), based on the academic thesis at `c:\git_repos\actuarial_agents`
(a 4-agent team — Data Preparation, Modelling, Reviewing, Explanation — orchestrated by a
Central Hub, built for motor/French-MTPL data with a locally-hosted HF model). The epic
(`AZD_AI_Agent_Epic_Claude.pdf`) asks for this reimagined with Claude as the LLM, several
explicit simplifications versus the thesis (no code generation, fixed model choice, drop
TCAV, self-reported confidence instead of token-logprobs), and stronger emphasis on
logging/memory/fairness for governance and regulatory approval.

Separately, AZD's AI architect has proposed a concrete delivery mechanism
(`potential_solution.svg`): ship this as a **Claude Code Plugin** that an actuary installs
and runs themselves in VS Code, on top of existing infra (GitHub Enterprise, Azure
Databricks/Unity Catalog/MLflow, Databricks Connect, ngdp-cli) — no new infra, no service
principal, actuary's personal identity throughout. This plan treats that architecture as
the primary delivery vehicle (verified against current Claude Code plugin/skill/marketplace
documentation) and maps the thesis's agent logic onto it, calling out where Claude Code's
actual primitives force a different design than the thesis's literal Python orchestrator.

The intended outcome: a document AZD can hand to the actuarial team and the architect that
(a) is technically grounded in what Claude Code plugins can/can't do, (b) reuses everything
reusable from the thesis, (c) is explicit about what must change for Claude/non-motor, and
(d) gives a clear phased path with named open dependencies rather than a hand-wavy "port it."

---

## 0. Verdict up front

**Adopt the architect's Claude-Code-plugin-on-existing-infra design as the primary delivery
vehicle**, with four concrete additions:
1. Treat "Central Hub" as **shared on-disk state (memory + audit log) plus a written
   routing runbook**, not a literal persistent orchestrator process — Claude Code has no
   long-lived cross-skill daemon.
2. Run the 4 phases as **skills invoked in one shared main-thread session**, not as 4
   independent subagents — subagents get an isolated context per invocation with no live
   back-reference to shared state (confirmed against Claude Code docs), which would break
   the thesis's `hub=self` cross-agent memory pattern.
3. Replace the thesis's token-logprob UProp/Bayesian-Network machinery with **self-reported
   confidence scores**, feeding the *same* BN structure via a bundled Python module.
4. Skip MCP for v1 — call Databricks Connect/MLflow Python SDKs directly via the Bash tool,
   already authenticated under the actuary's identity; revisit MCP only if the same tool
   surface needs exposing beyond this one plugin.

This was verified against current Claude Code documentation (plugin structure, `git-subdir`
marketplace sourcing for monorepos, hooks, subagent isolation) — not assumed.

### Why this, and not a standalone "real" agent solution?

The alternative would be to build a genuine standalone agent backend — e.g. literally port
`CentralHub` into a persistent service that calls the Claude API directly (own hosting,
own auth), or use Anthropic's separate **Managed Agents** product (server-hosted sessions,
a `multiagent: {type:"coordinator", agents:[...]}` config, its own credentialing and
scheduling). That is a materially heavier, more "correct-looking" architecture on paper,
but it is the wrong choice for this POC for concrete reasons:

- **New infra AZD explicitly wants to avoid.** A standalone service needs its own hosting
  decision, a service principal (GIAM/LIAM provisioning + approval cycle), its own CI/CD,
  monitoring, and an owning team — exactly what the epic's "no build capacity needed from
  the Data team" goal rules out for a POC.
- **Human-in-the-loop is structural, not bolted on.** Running inside the actuary's own
  interactive Claude Code session means every tool call, file write, and decision is visible
  and interruptible by construction. A backend agent service is architected to run
  unattended — which is a bigger governance ask to justify for a *first* POC on a
  regulated actuarial workflow, not a smaller one.
- **Auditability is native, not something to build.** Claude Code already logs every tool
  call in the session transcript and exposes hook points (`PostToolUse`) to capture them.
  The thesis had to hand-build `WorkflowAudit` from scratch for exactly this reason; a
  backend agent service would face the same build cost again.
- **Faster to build and to review.** Skills are markdown + a few vendored scripts —
  reviewable by an actuary directly (the epic's own "no coding framework to learn" framing).
  A backend service is code the Data/platform team must build, secure, and maintain.
- **Billing/ops simplicity.** Claude Code usage rides the actuary's existing seat; a Managed
  Agents deployment is a separate product surface with its own billing and scheduling setup.

**The one real thing given up:** no unattended/scheduled runs (§1, constraints). That is the
correct and only reason to eventually reach for Managed Agents or a custom service — treat
it as a deferred, named future decision if AZD's roadmap ever needs "run this weekly with no
actuary present," not a reason to over-build the POC today.

---

## 1. Claude Code primitives — verified mapping

| Thesis concept (file) | Claude Code primitive | Notes |
|---|---|---|
| 4 agent classes (`agents/*.py`) | 4 **Skills** — `skills/<name>/SKILL.md`, plain-language instructions | Matches epic's "no coding framework" framing |
| `CentralHub` object holding `current_metadata`, `agents.hub=self` (`agents/central_hub.py:16-61`) | No literal equivalent. Realized as: the shared main-thread conversation (within one run) + on-disk memory/audit files (durable across sessions) | Subagents are isolated per invocation (verified: "Each subagent runs in its own context window"); a `memory: project` field on a subagent gives a persistent *directory*, not live shared state — insufficient for `current_metadata` threading. **Decision: phases run as skills in the main thread, not as subagents.** |
| `run_workflow()` state machine, `MAX_ITERATIONS=4`, routing (`agents/central_hub.py:63-453`) | A `workflow-router` skill/runbook encoding `utils/decision_mapping.py`'s routing tables, backed by a deterministic `check_iteration_cap.py` script | No native loop/retry construct spans skills; looping happens by re-invoking a skill in the conversation. Enforcement is a script call, not "hope Claude remembers the cap." |
| `CentralMemory` (`utils/central_memory.py`, JSON file) | A file (or Unity Catalog table, see §4) that skill instructions read at start / append at end | Direct fit, no new infra required at MVP (flat file); flagged as a candidate upgrade |
| `WorkflowAudit`, `UncertaintyGraphBN` (`utils/audit.py`) | Vendored Python modules (`scripts/audit_log.py`, `scripts/uncertainty_bn.py`) invoked via Bash after each phase | pyagrum runs unmodified in the actuary's Python env; BN structure/CPT-update logic (`update_from_uprop`, `infer()`) carries over almost verbatim — only the scalar fed in changes (§3) |
| Guardrail modules (`utils/fairness_module.py`) | Vendored `scripts/fairness_module.py`, columns generalized | Direct fit |
| Sandboxed `exec()` of LLM-generated code (`data_prep_agent.py:52-136`, `modelling_agent.py`) | **Removed** — no code generation per epic | Skills call fixed pipeline functions directly |
| TCAV (`utils/tcav_module.py`, needs `get_hidden_states_for_texts`) | **Dropped** — no hosted-model equivalent, epic pre-approved this |  |

**Manifest/folder structure** (confirmed against current plugin docs), nested per the
architect's proposal, no separate repo:

```
ngdp-dph-pricing/assets/ai/pricing_global/risk_modelling_agent/   <- plugin root
  .claude-plugin/plugin.json
  skills/
    data-preparation/SKILL.md
    model-training/SKILL.md
    review/SKILL.md
    explanation/SKILL.md
    workflow-router/SKILL.md        <- "Central Hub" routing runbook
  scripts/
    uncertainty_bn.py                <- ported utils/audit.py::UncertaintyGraphBN
    fairness_module.py               <- ported + generalized utils/fairness_module.py
    consistency.py                   <- ported utils/consistency.py
    audit_log.py                     <- ported utils/audit.py::WorkflowAudit
    central_memory.py                <- ported utils/central_memory.py
    check_iteration_cap.py           <- new, deterministic guardrail
  hooks/hooks.json                   <- PostToolUse logging backstop
  README.md
```

This nests inside the existing `ngdp-dph-pricing` repo, referenced from a marketplace via
the **`git-subdir`** plugin source type (`{"source":"git-subdir","url":"...","path":"assets/ai/pricing_global/risk_modelling_agent"}`),
which does a sparse/partial clone — confirmed current and purpose-built for exactly this
monorepo-nesting case. No separate plugin repo is needed, confirming the architect's claim.

### 1.1 How the `workflow-router` skill + `check_iteration_cap.py` actually work

This is the piece that replaces `agents/central_hub.py::run_workflow()`'s `while` loop
(lines 89-341) — worth spelling out concretely since there's no Python loop underneath it
anymore, only conversation + files.

**What `run_workflow()` did mechanically:** a Python `while continue_workflow:` loop held
`phase` and `iteration` as real variables in memory, called the right agent, read the
decision string out of `current_metadata["action"]`, and used a plain `if/elif` chain
(lines 206-256, 289-340) to set the next `phase` or increment `iteration` and jump back —
all enforced by the interpreter, impossible to skip.

**What replaces it, step by step:**

1. **The routing table itself** — `utils/decision_mapping.py`'s `ROUTING_MAP_REVIEW` /
   `ROUTING_MAP_EXPLANATION` (decision string → next phase) gets copied almost verbatim into
   `skills/workflow-router/SKILL.md` as a literal table, e.g.:

   > After the Review skill finishes, read its `action` field from the run's metadata file.
   > - `proceed` → if the prior phase was Data Prep, run the Model Training skill next; if
   >   the prior phase was Modelling, run the Explanation skill next.
   > - `reclean_data` → run `scripts/check_iteration_cap.py`. If it returns `ok`, run Data
   >   Preparation again. If it returns `cap_reached`, stop and tell the actuary to escalate.
   > - `retrain_model` → same, but re-run Model Training.
   > - `abort_workflow` → stop immediately, write `status=terminated` to the audit log.

   This table is *instructions Claude follows*, not code — so on its own it is advisory:
   nothing stops Claude from "forgetting" to check the cap, the way `MAX_ITERATIONS` in the
   thesis mechanically could not be skipped.

2. **`check_iteration_cap.py` is what makes the cap actually deterministic.** It's a tiny,
   ordinary script (not an LLM call) that: reads the audit log for the current run, counts
   how many times the current phase has already been re-entered, and prints `ok` or
   `cap_reached` (mirroring the thesis's `if iteration >= MAX_ITERATIONS` check at
   `central_hub.py:223-227`/`237-241`/`307-311`/`320-324`). The `workflow-router` skill's
   instructions *require* Claude to run this script and act on its literal output before
   looping back — so the enforcement point moves from "the interpreter can't skip it" to
   "a deterministic script Claude is instructed to call and obey," which is materially
   different but effectively equivalent in practice as long as the instruction is followed.
   This is why the hook backstop (below) matters: it's a second, code-level check that the
   router skill's instructions were actually followed, not just written.

3. **Who actually "loops"?** In the thesis, the Python process loops by itself, no human
   involved. Here, after each skill finishes, either (a) Claude — following the
   `workflow-router` instructions in the same conversation — proposes and, if permitted,
   directly invokes the next skill, or (b) the actuary reads the routing decision Claude
   surfaces and explicitly runs the next skill themselves. Given the epic's human-in-the-loop
   principle, **recommend (b) as the default for the POC**: Claude states the decision and
   the recommended next step, the actuary confirms/triggers it. This is a deliberate, small
   loss of automation in exchange for a stronger human checkpoint at every phase transition —
   consistent with the epic's own preference to keep actuaries "in control of the workflow."

**Constraints to flag to AZD (discovered, not assumed):**
- No mechanical iteration-cap enforcement across skills — must be backed by
  `check_iteration_cap.py`, called from every phase's instructions.
- Fully unattended/scheduled runs are out of scope for this architecture (personal-identity,
  interactive-session model). If AZD later wants that, the correct primitive is Anthropic's
  separate Managed Agents product (server-hosted, own credentialing) — a distinct decision,
  not a Claude Code plugin feature.
- `PostToolUse` hooks (`hooks/hooks.json`, matcher `Bash|Write|Edit`) give a cheap, automatic
  logging backstop independent of whether a skill remembered to call the logging module —
  include from Phase 1.

---

## 2. Mapping thesis components → POC components (per epic deviation)

| Thesis file/mechanism | Epic deviation | POC design |
|---|---|---|
| Motor data, hard-coded columns (`utils/data_cleaning.py`, `utils/data_pipeline.py`) | Switch to Home/Household | New pipeline needed — **real open dependency**, not designed here (§4) |
| `data_prep_agent.py` layers 2-3 (LLM generates/verifies cleaning code) | No code generation | Removed. Data Preparation skill calls the existing validated cleaning function/notebook directly; only planning (layer 1) and summarizing (layer 4) remain as LLM reasoning |
| `modelling_agent.py` layer 1 (GLM/GBM choice) + layer 2 (LLM-generated training code) | No code gen; fixed model choice | Model Training skill hard-codes the model family (TBD, §4) and calls the existing trainer directly; metrics/impact analysis (layers 3-4) remain LLM reasoning |
| `review_agent.py` full layer structure incl. `utils/consistency.py` | Keep layered prompting + memory; consistency check "crucial" | Kept near-unchanged — this agent never did code-gen or model-choice |
| `explanation_agent.py` layer1 (belief), layer2 (TCAV), layer3 (fairness), layer4 (decision) | Belief optional-keep; TCAV drop; fairness must-keep+generalize; uncertainty via self-reported confidence | Belief kept as one extra prompt. TCAV deleted, no replacement. Fairness (`utils/fairness_module.py::group_fairness`) kept, columns generalized to Home/Household attributes (pending actuarial sign-off). Decision formula drops the TCAV term. |
| `utils/prompt_library.py::PROMPTS` | Keep layered prompting + short-term memory between layers | Moves into each skill's `SKILL.md` body; short-term memory between layers is the natural conversational continuity within one skill invocation |
| `utils/central_memory.py` | Keep/strengthen — essential | Ported; consider Unity Catalog table over flat JSON for multi-actuary/multi-run safety (§4) |
| Sandboxed `exec()` deterministic-fallback pattern (`_apply_llm_pipeline`) | Simplify meaning of "fallback" | New definition: if a skill's Databricks call fails or produces no valid structured output, log the failure and run the deterministic pipeline function with default parameters directly, skipping the LLM decision for that step — no adaptive path to compare against anymore |
| `agents/central_hub.py::run_workflow()` | Kept conceptually, delivery mechanism changes | Realized as main-thread skill sequencing + `workflow-router` skill encoding `utils/decision_mapping.py`'s routing tables + on-disk audit log as the durable iteration counter |
| `llms/wrappers.py::LLMWrapper` (HF transformers, Colab/Drive dependency) | Replace with Claude | **Mostly not needed at all** — inside a Claude Code skill, Claude Code's own runtime *is* "the LLM"; no separate API wrapper to build for the primary path (see §3.1) |

---

## 3. Claude-specific engineering gaps

### 3.1 No Anthropic API wrapper needed for the primary path
The thesis has zero Anthropic/OpenAI SDK usage. Because skills run inside the actuary's own
Claude Code session, Claude Code itself is the runtime — there is no separate LLM client to
build, eliminating `llms/wrappers.py`'s entire surface (HF model loading, Colab/Drive
bootstrap, `generate_n_samples`) as dead weight for this delivery model. A direct
`anthropic` SDK integration would only be needed for a future unattended/scheduled variant
(Managed Agents track) — explicitly deferred, not designed here.

**Is this consistent across multiple actuaries running the workflow?** Two separate
questions hide inside "consistent":

- **Model consistency (same prompts → comparably-behaved model):** yes, by construction.
  Every actuary's Claude Code session runs the same Claude model with the same skill
  instructions — there's no per-actuary model selection or fine-tuning to drift. This is
  actually *more* consistent than the thesis, which pinned one shared `LLMWrapper` instance
  per *run*, not globally.
- **Data/results consistency (does actuary A's run see actuary B's history, and do
  concurrent runs corrupt each other's state?):** this is the real risk, and it's a direct
  consequence of the central-memory storage choice (§4.7), not the LLM. A flat JSON file
  (the thesis's `data/memory/central_memory.json`) is **not safe** if two actuaries can run
  the workflow at the same time — concurrent read-modify-write on one file will silently
  lose updates (whoever writes last wins), and the Reviewing skill's consistency checks
  would then be comparing against an incomplete history. This is exactly why §4.7 flags
  moving central memory to a Unity Catalog table/Delta table as more than a nice-to-have:
  a table gives atomic appends and lets every actuary's run see the same consistent history
  regardless of timing. **Recommendation: if more than one actuary will realistically run
  this concurrently during the POC, treat the Unity Catalog table decision (§4.7) as
  required for Phase 1, not optional-later** — a flat file is fine only if the POC's actual
  usage pattern is "one actuary, one run at a time" during the pilot phase.

### 3.2 No logprobs → self-reported confidence, feeding the unchanged BN
Claude's API doesn't expose per-token logprobs the way local HF `generate(output_scores=True)`
does (`llms/wrappers.py:115-162`), so `utils/uprop.py`'s IU/EU split has no direct input.
Epic pre-approves the fix: self-reported confidence. Concrete pattern, once per skill at the
end of its final layer:

```
Before finishing this phase, self-assess your confidence (0.00-1.00) that this phase's
output is correct and safe to pass on. Consider data-quality issues noticed, whether the
pipeline ran without errors, whether a fallback was triggered, and how much you had to
extrapolate. Respond with:
Confidence: <0.00-1.00>
Rationale: <one sentence>
```

Parsed the same way the thesis extracts `Decision:` tokens, passed as `p_ok` directly into
`scripts/uncertainty_bn.py`'s CPT update (`utils/audit.py::UncertaintyGraphBN.update_from_uprop`,
lines 222-263) — **BN structure and `infer()` are unchanged**; only the scalar source
changes, no IU/EU decomposition needed. Satisfies acceptance criterion 5.

**Cost/latency guardrail:** do NOT replicate the thesis's 5x-sample-per-layer pattern
(`generate_n_samples(n=5)`, cheap on a local free model, not cheap on the Claude API across
~4-6 layers × 4 skills per run). One Claude turn per layer, self-reported confidence only;
reserve any resampling/self-consistency check for a specific, later-justified decision point
(e.g. only the final Review/Explanation decision), not as a default.

### 3.3 TCAV — confirmed drop, no replacement
`utils/tcav_module.py` needs hidden-state access fundamentally unavailable from a hosted
model. Epic already anticipated this ("only possible with open source models"). Fairness +
self-reported uncertainty are explicitly named as sufficient explainability coverage —
remove the TCAV call site and `tcav_score` term from the Explanation skill's decision logic.

### 3.4 Home/Household data pipeline — confirmed available, integration is the remaining work
`utils/data_cleaning.py`, `utils/data_pipeline.py` (hard-coded `VehAge`/`DrivAge`/`BonusMalus`/
`Area`/`ClaimNb`/`Exposure`/`IDpol`), `utils/fairness_module.py` (`DrivAge`/`Density` groups),
`utils/model_evaluation.py` (Poisson-frequency-specific metrics) are motor-frequency-specific
and are **not** ported as-is. **AZD has confirmed a validated Home pipeline already exists
in the Databricks estate**, which removes the biggest open dependency originally flagged
here (there is no need to build/validate a new pipeline from scratch for Phase 1). The
remaining, narrower work per skill:
- **Data Preparation skill**: call the existing Home pipeline's cleaning step directly
  instead of `utils/data_cleaning.py::DataCleaning.clean()` — needs the actual
  function/notebook name, its expected inputs/outputs, and default parameters (a short
  discovery task, not a design question).
- **Model Training skill**: same for the existing Home training pipeline in place of
  `GLMTrainer`/`GBMTrainer` — plus confirming which model family it already uses, which may
  settle the "fixed model choice" open question (§4.1) for free if the existing pipeline
  only supports one.
- **Fairness/evaluation**: `utils/fairness_module.py`'s `DrivAge`/`Density` groupings and
  `utils/model_evaluation.py`'s Poisson-frequency metrics still need Home-appropriate
  equivalents (target definition, protected attributes) — this part still needs actuarial
  input (§4.6), independent of the pipeline existing.
**Action for Phase 0/1: get pointed at the existing Home pipeline's code/notebook location
and interface before writing the Data Preparation skill**, rather than treating this as a
blocking unknown.

---

## 4. Open questions needing actuarial/architect input

1. **Fixed model choice: GLM or GBM?** — blocks Phase 2. May be settled for free once the
   existing Home pipeline's model family is confirmed (§3.4).
2. ~~Does a validated Home/Household pipeline exist?~~ **Resolved — yes, one exists.**
   Remaining task is discovery (function/notebook interface, params), not validation work.
3. **Databricks Connect + ngdp-cli invoked from inside a Claude Code Bash call** — likely
   works, needs an empirical smoke test in Phase 0 rather than being assumed.
4. **Cost estimate for Claude API calls per run** once real Home/Household prompt sizes
   exist (~16-24 Claude turns/run under the no-5x-sampling design, §3.2) — confirm with
   whoever owns AZD's Claude workspace billing.
5. **Fairness protected-attribute choices for Home/Household** (property age? construction
   type? flood-risk band? postcode-density proxy?) — needs actuarial/compliance sign-off.
6. **Central memory storage location** — flat JSON file (simplest, matches thesis, but unsafe
   under concurrent multi-actuary runs, §3.1) vs. a Unity Catalog/Delta table (atomic
   appends, safe under concurrency, a small explicitly-flagged exception to "no new infra")
   — required if more than one actuary will run this concurrently during the POC.

---

## 5. Phased build plan

| Phase | Scope | Deliverables | Acceptance criteria |
|---|---|---|---|
| **0 — Environment & access** | GIAM/Databricks (NGDP) access, VS Code + Databricks extension, Claude Code install, LIAM "Data Product Contributor" role, `ngdp-cli`; stand up the marketplace via `git-subdir`; smoke-test Databricks Connect from Claude's Bash tool | Working session with plugin installed; one verified Bash→Databricks Connect call | precondition |
| **1 — Data Preparation skill, end-to-end on Databricks** | Discover the existing Home pipeline's cleaning interface (§3.4); `skills/data-preparation/SKILL.md` calling it directly; wire `central_memory.py`, `audit_log.py`, `consistency.py`; deterministic-fallback clause; decide central-memory storage (flat file vs. table, §4.6) | One phase runs fully, logs to memory + audit trail | 1 (partial), 2, 3, 7 |
| **2 — Add Model Training + Review, basic routing** | Discover existing Home training pipeline's interface; `model-training/SKILL.md` (fixed model per §3.4/§4.1) + `review/SKILL.md` (full layers incl. consistency check); `workflow-router` skill; `check_iteration_cap.py` | End-to-end DataPrep→Model→Review loop with visible approve/retrain/reclean routing | 1, 2, 3, 7 |
| **3 — Add Explanation skill; strengthen logging** | `explanation/SKILL.md` (belief + fairness + decision, no TCAV); `PostToolUse` audit hook | Full 4-phase workflow; structured fairness report every run | 1, 2, 3, 4, 6, 7 |
| **4 — Uncertainty/confidence guardrail** | Self-reported confidence prompt in every skill layer; `uncertainty_bn.py` invoked after each phase; workflow-level posterior surfaced in the final report | Per-agent confidence propagated via BN to a workflow-level metric per run | 5 |
| **5 — Pilot with a real actuary** | Full workflow on real/realistic data; feedback on report readability, fairness usefulness, confidence calibration; decide on the optional feedback-loop mechanism | Actuary sign-off; go/no-go | 4, 6, overall |

Blocking open questions repeated: fixed model choice (Phase 2, possibly resolved by pipeline
discovery), Databricks-Connect-from-Bash verification (Phase 0), cost estimate (Phase 1),
fairness attributes (Phase 3), memory storage/concurrency decision (Phase 1).

---

## 6. Verdict on the architect's proposed architecture (Task 2)

**Satisfies the epic's goals well:**
- Human-in-the-loop: strong — actuary drives every phase interactively, nothing runs
  unattended by default, better fit than a backend service for AZD's risk-mitigation stance.
- Auditability: yes, with this plan's additions (vendored audit/BN/fairness scripts +
  `PostToolUse` hook backstop) — the architect's own "Claude appends to a local report file"
  framing is directionally right but needed the concrete script/hook design added.
- No new infra / no separate plugin repo: **confirmed technically correct** via
  `git-subdir` marketplace sourcing, purpose-built for monorepo nesting.
- Explainability/governance (fairness + uncertainty): achievable identically to the thesis's
  local-Python approach — the modules run the same whether invoked by a local script or by
  Claude's Bash tool.

**Real risks to flag:**
- Personal-identity-only access (no service principal anywhere) is correct for a
  human-in-the-loop POC, but forecloses scheduled/unattended runs without a design change —
  name this explicitly as a deferred, out-of-scope decision if AZD's roadmap ever wants it.
- "Central Hub" is not a literal persistent object; iteration-cap/routing enforcement is
  advisory unless backed by `check_iteration_cap.py` (§1) — a real behavioral difference
  from the thesis's mechanical `while` loop that must be communicated, not glossed over.
- Central-memory storage as a flat file is unsafe if multiple actuaries run concurrently
  (§3.1) — resolve this explicitly rather than defaulting to the thesis's flat-JSON pattern.
- MCP was considered and deliberately deferred (not overlooked) — Bash + already-authenticated
  Databricks Connect is simpler and sufficient for one plugin; MCP earns its complexity only
  if the same tool surface needs exposing to other products.

**Recommendation:** adopt as-is, with the four additions in §0/§1 (vendored modules,
`PostToolUse` hook, `check_iteration_cap.py`, and treating central-memory storage location
as an explicit small decision rather than an assumption).

---

## 7. Implementation time & effort estimate

**Assumptions** (state these alongside the estimate — they drive it more than the engineering itself):
- 1 engineer building the plugin, with part-time actuarial input for fairness/target-definition
  decisions and periodic review, not a dedicated actuary embedded full-time until Phase 5.
- The confirmed existing Home pipeline (§3.4) has a usable, documented interface once located —
  if it turns out to need non-trivial wrapping (e.g. no callable function, only a notebook with
  hard-coded paths), Phase 1/2 estimates below should be treated as a floor, not a ceiling.
- Access/approval lead times (GIAM, LIAM, compute policy) are organizational processes outside
  engineering control — these are called out separately as elapsed-time risk, not effort.
- "Effort" = person-days of hands-on engineering work. "Elapsed" = realistic calendar time
  including review cycles, actuarial sign-off waits, and access provisioning lag.

| Phase | Effort (person-days) | Elapsed (calendar) | What drives the gap between effort and elapsed |
|---|---|---|---|
| **0 — Environment & access** | 1–2 | 1–3 weeks | Almost entirely GIAM/LIAM access-request lead time and compute-policy setup, not engineering — historically the least predictable phase on calendar time despite trivial effort |
| **1 — Data Preparation skill** | 5–8 | 1.5–2.5 weeks | First-time build of `central_memory.py`, `audit_log.py`, `consistency.py`, `check_iteration_cap.py` (reused unchanged in later phases) plus discovery of the existing Home pipeline's actual interface; the central-memory storage decision (§4.6) needs a Data-team answer before this phase can close |
| **2 — Model Training + Review + routing** | 5–8 | 1.5–2 weeks | Reuses Phase 1's scripts; new work is the Review skill's 5-layer structure (ported from `review_agent.py`) and the `workflow-router` skill; elapsed time mostly engineering, less external dependency |
| **3 — Explanation skill + logging hardening** | 4–6 | 1.5–3 weeks | Engineering itself is modest (belief + decision layers, `PostToolUse` hook); elapsed time dominated by waiting on fairness protected-attribute sign-off (§4.5), which is a compliance/actuarial decision outside engineering's control |
| **4 — Uncertainty/confidence guardrail** | 3–5 | 1 week | Porting `uncertainty_bn.py` is close to mechanical (§3.2 confirms structure is unchanged); most of the time is prompt-tuning the self-reported-confidence instructions and validating the BN posterior looks sane end-to-end |
| **5 — Pilot with a real actuary** | 3–5 (engineering) + actuary's own time | 2–3 weeks | Elapsed time is set by actuary availability and iteration on feedback, not engineering effort — treat this as schedule-driven, not estimate-driven |

**Total: roughly 21–34 person-days of engineering effort (~4–7 weeks of one engineer's
time), spread over an estimated 8–14 weeks of elapsed calendar time** given realistic access
lead times and sign-off waits. The elapsed-time range is wide because three of the six phases
(0, 3, 5) are gated by non-engineering dependencies (access provisioning, compliance sign-off,
actuary availability) that this plan cannot estimate precisely — they should be tracked as
schedule risks, not folded into a false-precision single number.

**What would compress this:**
- Resolving the fixed-model-choice and fairness-attribute questions (§4.1, §4.5) *before*
  Phase 1 starts, rather than discovering them mid-phase, removes the biggest source of
  rework risk in Phases 1–3.
- Confirming the existing Home pipeline's interface (§3.4) in Phase 0 rather than Phase 1
  turns a "discovery + integration" task into "integration only."
- A second engineer could parallelize Phase 1's script-porting work against skill-writing,
  but the phases are otherwise sequential by design (each depends on the previous skill
  existing) and don't parallelize well beyond that.

**What would extend this:** if the existing Home pipeline (§3.4) needs non-trivial rework to
be callable from a skill (e.g., currently only runs as a manually-parameterized notebook),
or if the central-memory storage decision (§4.6) lands on a Unity Catalog table requiring
Data-team provisioning rather than a flat file, add roughly 3–5 person-days and 1–2 elapsed
weeks to Phase 1 specifically.

---

## Critical files for implementation (thesis repo, for porting reference)

- `utils/audit.py` → port `UncertaintyGraphBN`, `WorkflowAudit` into `scripts/uncertainty_bn.py`, `scripts/audit_log.py`
- `utils/central_memory.py` → `scripts/central_memory.py`
- `utils/consistency.py` → `scripts/consistency.py` (Review skill's drift-check layer)
- `utils/fairness_module.py` → `scripts/fairness_module.py`, columns generalized
- `utils/prompt_library.py` → source material for each skill's `SKILL.md` prompt content
- `agents/central_hub.py` + `utils/decision_mapping.py` → source material for the `workflow-router` skill's routing logic
