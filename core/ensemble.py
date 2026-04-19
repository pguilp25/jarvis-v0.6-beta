"""
Ensemble — runs 2-5 AIs ALL IN PARALLEL. Groq + NVIDIA fire simultaneously.
At ~5 RPM real usage, we're nowhere near the 40 RPM limit.
"""

import asyncio
from core.retry import call_with_retry
from core.model_selector import select_domain_pair, select_for_context
from core.cli import step, status


async def run_ensemble(
    query: str,
    context: str,
    domain: str,
    complexity: int,
    context_tokens: int,
    assumption_prompt: str = "",
    last_exchange: str = "",
) -> list[dict]:
    """
    Run multiple AIs on the same query IN PARALLEL.
    Complexity >= 7: 2-step thinking (plan → verify) to avoid 504 timeouts.
    Returns list of {model, answer}.
    """
    step(f"Ensemble — domain={domain}, complexity={complexity}, ctx={context_tokens}")

    if complexity >= 7:
        # 2-STEP THINKING — each step is lighter, avoids 504 timeouts
        return await _two_step_ensemble(query, context, domain, complexity, context_tokens, assumption_prompt, last_exchange)

    # Simple single-prompt for lower complexity
    full_prompt = _build_prompt(query, context, assumption_prompt, last_exchange, complexity=complexity)

    if context_tokens < 8000:
        results = await _mixed_ensemble_small(full_prompt, domain)
    elif context_tokens < 25000:
        results = await _mixed_ensemble_medium(full_prompt, domain)
    else:
        results = await _nvidia_ensemble_3(full_prompt, domain, context_tokens)

    status(f"Got {len(results)} answers")
    return results


async def _two_step_ensemble(
    query: str, context: str, domain: str, complexity: int,
    context_tokens: int, assumption_prompt: str, last_exchange: str,
) -> list[dict]:
    """
    2-step thinking for complexity >= 7.
    Step A: Plan + hypotheses (all parallel, lighter calls)
    Step B: Verify + write final answer (all parallel, uses Step A output)
    """
    # Build Step A prompt
    step_a_prompt = _build_step_a(query, context, assumption_prompt, last_exchange)

    # Select models
    if complexity >= 9:
        models = [
            "nvidia/deepseek-v3.2",
            "nvidia/glm-5",
            "nvidia/minimax-m2.5",
            "nvidia/qwen-3.5",
            "nvidia/nemotron-super",
        ]
    else:
        pair = select_domain_pair(domain)
        models = list(pair)

    # Step A: All models plan in parallel
    step("Step A: Planning + hypotheses")
    step_a_results = list(await asyncio.gather(*[
        _call_one(m, step_a_prompt) for m in models
    ]))
    status(f"Step A done: {len(step_a_results)} plans")

    # Step B: Each model verifies ITS OWN plan and writes final answer
    step("Step B: Verify + write answer")
    step_b_tasks = []
    for plan in step_a_results:
        step_b_prompt = _build_step_b(query, plan["answer"])
        step_b_tasks.append(_call_one(plan["model"], step_b_prompt))

    results = list(await asyncio.gather(*step_b_tasks))
    status(f"Step B done: {len(results)} verified answers")
    return results


def _build_prompt(query: str, context: str, assumption_prompt: str, last_exchange: str = "", complexity: int = 5) -> str:
    """Build prompt. For complexity < 7, single prompt. For >= 7, use _build_step_a/b instead."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    from core.agent_context import get_agent_context
    agent_name = "chat_very_intelligent" if complexity >= 7 else "chat_intelligent"
    parts = [get_agent_context(agent_name), "", SYSTEM_KNOWLEDGE, ""]
    if context:
        parts.append(f"CONVERSATION CONTEXT:\n{context}\n")
    if last_exchange:
        parts.append(f"LAST EXCHANGE:\n{last_exchange}\n")
    if assumption_prompt:
        parts.append(f"ANALYSIS REQUIREMENTS:\n{assumption_prompt}\n")

    parts.append("""INSTRUCTIONS:

CONTEXT IS CRITICAL:
- BEFORE answering, check: does the user's message make sense ON ITS OWN?
- If it's ambiguous or uses words like "it", "that", "this", "more", "also",
  "another", "same", "again", "continue", "what about" — it REQUIRES the
  conversation context above.
- Re-read the LAST EXCHANGE. Figure out exactly what the user is referring to.
  State it explicitly in your answer.
- Do NOT drift to a new topic. Stay on the thread of the conversation.
- Do NOT answer a different question than what they're asking about.

ANSWERING:
- FIRST: Restate what the user is actually asking for in your own words.
  If the prompt is vague, infer using the conversation context.
- Read the USER QUERY below carefully — address EVERY point.
- Provide a thorough, well-reasoned answer.
- TOOL: If you need current info, write [WEBSEARCH: your query] on its own line.

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 short bullet points noting: what topic was discussed, any subject changes, key terms.""")

    parts.append(f"""
════════════════════════════════════════
USER QUERY (read this completely — including ALL numbered items, constraints, and rules):
════════════════════════════════════════
{query}
════════════════════════════════════════""")

    return "\n".join(parts)


# ─── 2-Step Thinking (complexity >= 7) ───────────────────────────────────────

def _build_step_a(query: str, context: str, assumption_prompt: str, last_exchange: str = "") -> str:
    """Step A: Plan + Define Hypotheses. Lighter call — no full answer needed."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    from core.agent_context import get_agent_context
    parts = [get_agent_context("chat_very_intelligent"), "", SYSTEM_KNOWLEDGE, ""]
    if context:
        parts.append(f"CONVERSATION CONTEXT:\n{context}\n")
    if last_exchange:
        parts.append(f"LAST EXCHANGE:\n{last_exchange}\n")
    if assumption_prompt:
        parts.append(f"ANALYSIS REQUIREMENTS:\n{assumption_prompt}\n")

    parts.append(f"""
════════════════════════════════════════
USER QUERY (read completely — ALL constraints and rules):
════════════════════════════════════════
{query}
════════════════════════════════════════

YOUR TASK — PLANNING PHASE ONLY (do NOT write the full answer yet):

0. INFER INTENT: Restate what the user ACTUALLY wants in your own words.
   If the query is ambiguous or uses words like "it", "that", "this", "more",
   "also" — check the CONVERSATION CONTEXT and LAST EXCHANGE above.
   The user is continuing a conversation. Their question builds on what came before.
   Do NOT interpret the query in isolation if context exists.

1. CONSTRAINTS: List EVERY constraint, requirement, and condition from the query above.
   Number them. Only what the user ACTUALLY asked for.

2. APPROACH: For each constraint, briefly explain how you would address it.

3. HYPOTHESES: Based on your analysis, state your key conclusions/recommendations as hypotheses:
   H1: [your first key claim]
   H2: [your second key claim]
   H3: [etc.]
   
   For each hypothesis, note what EVIDENCE would support or refute it.

4. RISKS: What could go wrong? What failure modes exist? What tradeoffs are you making?

5. KEY DATA: Include any calculations, numbers, specific values, or references that the 
   verification step will need. Do NOT leave anything out — the next step only sees YOUR output.

IMPORTANT: Include EVERYTHING the next step needs. Be thorough in your planning.""")

    return "\n".join(parts)


def _build_step_b(query: str, step_a_output: str) -> str:
    """Step B: Verify hypotheses + write final answer. Uses Step A output."""
    from core.system_knowledge import SYSTEM_KNOWLEDGE
    return f"""{SYSTEM_KNOWLEDGE}

You are in the VERIFICATION & WRITING phase. A planning AI already analyzed the query 
and produced hypotheses. Your job:

1. READ the original query and the planning output below
2. VERIFY each hypothesis — is it correct? Does it address the constraint?
3. WRITE the complete, thorough final answer

════════════════════════════════════════
ORIGINAL QUERY:
════════════════════════════════════════
{query}
════════════════════════════════════════

PLANNING AI's OUTPUT (hypotheses + approach):
{step_a_output}

YOUR TASK:
0. INFER INTENT: Before verifying, restate what the user ACTUALLY wants in your own words.
   This ensures your answer addresses their real goal, not just the literal words.

1. VERIFY each hypothesis:
   ✓ = correct, supported by evidence/logic
   ✗ = wrong or incomplete → FIX IT with correct information
   
2. CHECK: Does the plan address ALL constraints from the query? If any are missing, add them.

3. WRITE YOUR COMPLETE ANSWER — incorporate all verified hypotheses, corrected errors, 
   and missing constraints. Be thorough.

AFTER your answer, on a NEW line write exactly:
[CONTEXT_NOTES]
Then 1-3 short bullet points noting what was discussed."""


async def _call_one(model: str, prompt: str, max_tokens: int = 16384) -> dict:
    """Call one model with mid-thought web search. Used by gather()."""
    from core.tool_call import call_with_tools
    return await call_with_tools(
        model, prompt, project_root=None, max_tokens=max_tokens,
        enable_code_search=False, enable_web_search=True,
    )


async def _mixed_ensemble_small(prompt: str, domain: str) -> list[dict]:
    """< 8K context: 2 Groq + 1 NVIDIA — ALL parallel."""
    pair = select_domain_pair(domain)
    tasks = [
        _call_one("groq/kimi-k2", prompt),
        _call_one("groq/gpt-oss-120b", prompt),
        _call_one(pair[0], prompt),
    ]
    return list(await asyncio.gather(*tasks))


async def _mixed_ensemble_medium(prompt: str, domain: str) -> list[dict]:
    """8-25K context: 1 Groq + 2 NVIDIA — ALL parallel."""
    pair = select_domain_pair(domain)
    tasks = [
        _call_one("groq/llama-4-scout", prompt),
        _call_one(pair[0], prompt),
        _call_one(pair[1], prompt),
    ]
    return list(await asyncio.gather(*tasks))


async def _nvidia_ensemble_3(prompt: str, domain: str, ctx_tokens: int) -> list[dict]:
    """25-72K context: 3 NVIDIA — ALL parallel."""
    available = select_for_context(ctx_tokens, "medium")[:3]
    if len(available) < 3:
        available = select_for_context(ctx_tokens, "simple")[:3]

    tasks = [_call_one(m, prompt) for m in available]
    return list(await asyncio.gather(*tasks))


async def _nvidia_pair(prompt: str, pair: tuple) -> list[dict]:
    """2 NVIDIA domain-matched — parallel."""
    tasks = [_call_one(m, prompt, max_tokens=16384) for m in pair]
    return list(await asyncio.gather(*tasks))


async def _nvidia_ensemble_5(prompt: str, ctx_tokens: int) -> list[dict]:
    """Full 5-model ensemble — ALL parallel."""
    all_nvidia = [
        "nvidia/deepseek-v3.2",
        "nvidia/glm-5",
        "nvidia/minimax-m2.5",
        "nvidia/qwen-3.5",
        "nvidia/nemotron-super",
    ]
    available = [m for m in all_nvidia if m in select_for_context(ctx_tokens, "extreme")]
    if len(available) < 3:
        available = [m for m in all_nvidia if m in select_for_context(ctx_tokens, "hard")]
    tasks = [_call_one(m, prompt, max_tokens=16384) for m in available]
    return list(await asyncio.gather(*tasks))


# ─── Two-Phase Racing System ──────────────────────────────────────────────────

async def run_two_phase_race(
    planning_models: list[str],
    revision_models: list[str],
    planning_prompt: str,
    base_revision_prompt: str,
    original_query: str,
    context: str,
    n_winners: int = 3,
) -> tuple[list[dict], list[dict]]:
    """
    Two-phase racing system with disjoint model pools.

    Phase 1 (Planning Race): 4 AIs race to create plans → keep first 3 to finish.
    Phase 2 (Revision Race): 4 DIFFERENT AIs race to revise the 3 plans → keep first 3 to finish.

    Returns (planning_results, revision_results) where each is a list of n_winners dicts.
    """
    from core.cli import step, status, warn

    # Validate model pools are disjoint
    overlap = set(planning_models) & set(revision_models)
    if overlap:
        warn(f"Model pool overlap detected: {overlap}. Removing duplicates from revision pool.")
        revision_models = [m for m in revision_models if m not in overlap]

    if len(planning_models) < n_winners:
        raise ValueError(f"Need at least {n_winners} planning models, got {len(planning_models)}")
    if len(revision_models) < n_winners:
        raise ValueError(f"Need at least {n_winners} revision models, got {len(revision_models)}")

    # ── Phase 1: Planning Race ───────────────────────────────────────────────
    step("Phase 1: Planning race — 4 models parallel")
    planning_calls = [(m, planning_prompt) for m in planning_models[:4]]
    planning_results = await _race_first_n(planning_calls, n=n_winners)

    if not planning_results:
        warn("All planning models failed — falling back to single model")
        fallback = await _call_one("nvidia/deepseek-v3.2", planning_prompt)
        planning_results = [fallback]

    # Filter out empty/invalid plans
    valid_plans = []
    for result in planning_results:
        if result.get("answer") and len(result["answer"].strip()) > 50:
            valid_plans.append(result)
        else:
            warn(f"Invalid plan from {result.get('model', 'unknown')}: empty or too short")

    if not valid_plans:
        warn("No valid plans — using fallback")
        fallback = await _call_one("nvidia/deepseek-v3.2", planning_prompt)
        valid_plans = [fallback]

    status(f"Phase 1 complete: {len(valid_plans)} valid plans from {[p['model'].split('/')[-1] for p in valid_plans]}")

    # ── Phase 2: Revision Race ───────────────────────────────────────────────
    step("Phase 2: Revision race — 4 models parallel")

    # Create revision prompts for each valid plan
    revision_prompts = _create_revision_prompts(original_query, context, valid_plans, base_revision_prompt)

    # For each plan, we race all 4 revision models
    # We keep the first 3 revisions to complete across ALL revision tasks
    revision_calls = []
    for i, rev_prompt in enumerate(revision_prompts):
        for model in revision_models[:4]:
            revision_calls.append((f"{model}::plan{i}", rev_prompt))

    # Race all revision tasks, keeping first n_winners
    revision_results = await _race_first_n(revision_calls, n=n_winners)

    if not revision_results:
        warn("All revision models failed — returning planning results unchanged")
        return valid_plans, valid_plans

    # Clean up model names (remove ::planX suffix)
    for result in revision_results:
        if "::" in result.get("model", ""):
            result["model"] = result["model"].split("::")[0]

    status(f"Phase 2 complete: {len(revision_results)} revisions from {[r['model'].split('/')[-1] for r in revision_results]}")

    return valid_plans, revision_results


def _create_revision_prompts(
    original_query: str,
    context: str,
    planning_results: list[dict],
    base_revision_prompt: str,
) -> list[str]:
    """
    Create revision prompts from planning results.
    Each prompt asks a revision AI to review and improve one plan.
    """
    from core.system_knowledge import SYSTEM_KNOWLEDGE

    revision_prompts = []

    for i, plan in enumerate(planning_results):
        plan_model = plan.get("model", "unknown").split("/")[-1]
        plan_content = plan.get("answer", "")

        # Truncate if too long (preserve key sections)
        max_plan_len = 15000
        if len(plan_content) > max_plan_len:
            # Try to preserve structure by keeping first and last portions
            half = max_plan_len // 2
            plan_content = plan_content[:half] + "\n\n[...middle section truncated for length...]\n\n" + plan_content[-half:]

        prompt = f"""{SYSTEM_KNOWLEDGE}

You are in the REVISION phase. A planning AI has already created an implementation plan.
Your job: Review, critique, and IMPROVE this plan.

════════════════════════════════════════
ORIGINAL REQUEST:
════════════════════════════════════════
{original_query}

════════════════════════════════════════
CONTEXT:
════════════════════════════════════════
{context[:5000] if context else 'No additional context provided.'}

════════════════════════════════════════
PLANNING AI's OUTPUT ({plan_model}):
════════════════════════════════════════
{plan_content}

════════════════════════════════════════
YOUR TASK — REVISE AND IMPROVE:
════════════════════════════════════════
1. REVIEW: Identify any flaws, missing details, or unclear sections in the plan above.
2. FIX: Correct errors, fill gaps, and clarify ambiguities.
3. ENHANCE: Add missing edge cases, error handling, and implementation details.
4. VALIDATE: Ensure the plan addresses ALL requirements from the original request.
5. OUTPUT: Write the COMPLETE, IMPROVED implementation plan.

{base_revision_prompt}

IMPORTANT: Output the full revised plan, not just changes. The revised plan should be complete and ready for implementation."""

        revision_prompts.append(prompt)

    return revision_prompts


async def _race_first_n(calls: list[tuple[str, str]], n: int = 2) -> list[dict]:
    """
    Fire all models in parallel, return the first N to finish. Cancel the rest.
    Each call is a tuple of (model, prompt).
    """
    from core.cli import status

    results = []
    pending = set()

    async def _run(model, prompt):
        return await _call_one(model, prompt)

    for model, prompt in calls:
        task = asyncio.create_task(_run(model, prompt))
        task.model_id = model
        pending.add(task)

    while len(results) < n and pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            try:
                result = task.result()
                results.append(result)
            except Exception as e:
                status(f"⚠️ {task.model_id} failed: {e}", "yellow")

    # Cancel remaining tasks
    for task in pending:
        task.cancel()
        status(f"Dropped slowest: {task.model_id}")

    return results[:n]
