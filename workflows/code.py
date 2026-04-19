"""
Coding Agent -- JARVIS v0.5.1

Workflow:
  Phase 2 -- PLAN:
    Standard (complexity < 7):
      Layer 1: 4 AIs write independent plans (parallel)
      Layer 2: GLM-5 merges best from all into final plan
    Deep (complexity >= 7 or !!deepcode):
      Layer 1: 4 AIs write independent plans (parallel)
      Layer 2: 4 AIs each read ALL plans, find flaws/strengths,
               write improved plan (parallel)
      Layer 3: GLM-5 reads all 4 improved plans, finds flaws/strengths,
               writes THE final plan
  Phase 3 -- IMPLEMENT: GLM-5 codes each file in parallel
  Phase 3.5 -- REVIEW: GLM-5 reviews each file in parallel,
               finds flaws, writes fixes
  Phase 4 -- TEST: optional (only if user asks)
  Phase 5 -- DELIVER: show diff, ask to apply
"""

import asyncio
import os
import re
import subprocess
from core.retry import call_with_retry
from core.tool_call import call_with_tools as _call_with_tools
from core.state import AgentState
from core.cli import step, status, success, warn, error
from core.system_knowledge import SYSTEM_KNOWLEDGE
from clients.gemini import call_flash
from tools.codebase import (
    scan_project, read_file, read_files, search_code,
    format_search_results, run_on_demand_searches,
    extract_search_requests,
)
from tools.sandbox import Sandbox


# ─── Models ──────────────────────────────────────────────────────────────────

UNDERSTAND_MODELS = [
    "nvidia/deepseek-v3.2",
    "nvidia/qwen-3.5",
    "nvidia/minimax-m2.5",
]

IMPLEMENT_MODEL = "nvidia/glm-5"

NVIDIA_5 = [
    "nvidia/deepseek-v3.2",
    "nvidia/glm-5",
    "nvidia/minimax-m2.5",
    "nvidia/qwen-3.5",
    "nvidia/nemotron-super",
]

NVIDIA_3 = [
    "nvidia/deepseek-v3.2",
    "nvidia/qwen-3.5",
    "nvidia/minimax-m2.5",
]


# ─── Prompts ─────────────────────────────────────────────────────────────────

from core.agent_context import get_agent_context as _get_agent_context

UNDERSTAND_PROMPT = _get_agent_context("code") + "\n\n" + SYSTEM_KNOWLEDGE + """

You are a code analyst. Your job: find ALL relevant code for this task.

CONTEXT IS CRITICAL:
If the task is ambiguous or references something from conversation ("fix that bug",
"add the feature we discussed", "do the same for X"), use the conversation context
to understand what the user actually wants. Do NOT guess — look at what was discussed.

TASK: {task}

PROJECT STRUCTURE:
{project_structure}

YOUR JOB:
1. Restate what the user wants in your own words
2. Look at the project structure — which files are relevant?
3. Search for the specific code you need to understand — escalate in order:
   [REFS: name]           — find definitions, imports, usages (start here)
   [LSP: name]            — semantic deps, types (if REFS not enough)
   [SEARCH: pattern]      — ripgrep pattern search
   Write only the tags you need, then STOP. No duplicates.

Output:
## RELEVANT FILES
- path/to/file.py — why it's relevant

## SEARCH REQUESTS
[REFS: the_key_function]

## KEY CODE SECTIONS
- function_name in file.py — what it does, why it matters

## TASK CLASSIFICATION
- bug fix / new feature / refactor / optimization
- estimated complexity: simple / medium / complex
"""

PLAN_COT_EXISTING = """
══════════════════════════════════════════════════════════════════════
YOUR CHAIN OF THOUGHT — EXPAND LEVEL BY LEVEL
══════════════════════════════════════════════════════════════════════

You MUST think by EXPANDING, level by level. Never jump ahead.
At each level, check: do I need to zoom in further? Use tools to expand.

┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — UNDERSTAND THE TASK                                   │
└─────────────────────────────────────────────────────────────────┘

Read the TASK and the PROJECT OVERVIEW above.

Write your thinking:
  - What is the user asking for? Restate in your own words.
  - Separate CONTEXT (what exists now) from INSTRUCTIONS (what to change).
    Not everything in the prompt is a request — some of it describes the current state.
  - Which files does this touch? (use the general map to identify them)
  - What do I already know from the maps above?
  - What do I still need to look up?

If the user's request is ambiguous ("it", "that", "this"), check the
CONVERSATION CONTEXT — they're continuing a previous discussion.

┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — LOOK UP WHAT YOU NEED (only if needed)                │
└─────────────────────────────────────────────────────────────────┘

If you need more info to plan, escalate through the tools in order:

  1. [REFS: name]         — start here: find definitions, imports, usages (fast)
  2. [LSP: name]          — if REFS wasn't enough: semantic deps, types
  3. [DETAIL: section]    — if still not enough: organized code map for a feature
     [PURPOSE: category]  — all code for a purpose (e.g. "API calls", "UI colors")
  4. [CODE: path/file]    — last resort: read the actual source code

Write ONLY the tags you need, then STOP. Do NOT write duplicate tags.
If the general map already tells you enough, skip this step entirely.

┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — WRITE THE PLAN                                        │
└─────────────────────────────────────────────────────────────────┘

Write the plan using the format below.
If you realize mid-plan you need one more lookup, write the tag and STOP.

══════════════════════════════════════════════════════════════════════
"""

PLAN_COT_NEW = """
══════════════════════════════════════════════════════════════════════
YOUR CHAIN OF THOUGHT — EXPAND LEVEL BY LEVEL
══════════════════════════════════════════════════════════════════════

This is a NEW project — there is no existing code. Do NOT use [DETAIL:],
[CODE:], or [SEARCH:] tags — there is nothing to look up.

You MUST think by EXPANDING, level by level. Never jump ahead.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 1 — GENERAL ARCHITECTURE                                │
└─────────────────────────────────────────────────────────────────┘

What is the big picture? Write your thinking:
  - What is the user asking for? Restate in your own words.
  - What files/modules will this project have?
  - How do they connect? What is the data flow between them?
  - What is the user-facing behavior at a high level?

Do NOT go into detail yet. Just the skeleton.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 2 — EXPAND EACH SECTION INTO DETAIL                     │
└─────────────────────────────────────────────────────────────────┘

Now take EACH file/module from Level 1 and expand it:
  - What functions/classes does it contain?
  - What data structures and state does it manage?
  - What is the exact behavior of each function? (inputs, outputs, logic)
  - What edge cases exist?
  - How does it interact with the other modules?

Go through every section. Do not skip any.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 3 — WRITE THE PLAN                                      │
└─────────────────────────────────────────────────────────────────┘

Now that you have expanded everything, write the final plan using the
format below. The plan should be detailed enough that a developer can
implement each file without asking questions.

You can use [WEBSEARCH: query] if you need to look up an API, library,
or technique you're not sure about.

══════════════════════════════════════════════════════════════════════
"""

PLAN_PROMPT = SYSTEM_KNOWLEDGE + """

══════════════════════════════════════════════════════════════════════
YOUR ROLE: PLANNER
══════════════════════════════════════════════════════════════════════
You are a software architect. Your ONLY job is to write an implementation plan.

WHY YOU EXIST: You are one of several AIs writing independent plans.
After you finish, your plan will be compared against other plans by another
set of AIs who will find flaws, pick the best ideas, and merge everything
into one final plan. That final plan gets handed to a SEPARATE coding AI
(GLM-5) that implements it. If your plan is vague, the coder will guess
wrong. If your plan has logic errors, the reviewer might miss them.
BE PRECISE. BE EXPLICIT. Cover edge cases.

YOU DO:
  - Analyze the task and the codebase
  - Design the solution — what to change, where, and how the logic works
  - Describe behavior, data flow, edge cases in plain English
  - Use tools to understand the existing code before planning

YOU DO NOT:
  - Write ANY code — no snippets, no pseudo-code, no function calls
  - Implement anything — that is a different AI's job

TOOLS — use in this order, escalate only if you need more info:
  1. [REFS: name]           — find all definitions, imports, usages (fast, ripgrep)
  2. [LSP: name]            — semantic search: dependencies, types, indirect refs
                              (only if REFS didn't give enough)
  3. [DETAIL: section name] — organized code map for a feature
     [PURPOSE: category]    — all code serving a purpose (e.g. "API calls", "UI colors")
  4. [CODE: path/to/file]   — read actual source code (last resort — full file)
     [SEARCH: pattern]      — ripgrep pattern search
     [WEBSEARCH: query]     — web search for API docs, libraries

Write all the tags you need, then write STOP on its own line.
══════════════════════════════════════════════════════════════════════

TASK: {task}

PROJECT OVERVIEW (general map):
{context}

{cot_instructions}

PLAN FORMAT — write a DETAILED plan in PLAIN ENGLISH. NO CODE ANYWHERE.
══════════════════════════════════════════════════════════════════════
The plan must be detailed enough that a developer can implement it WITHOUT
asking questions. But it must contain ZERO code — no snippets, no pseudo-code,
no function calls, no HTML tags, no CSS properties, no variable assignments.

Describe WHAT to build and HOW it should work, not the code itself.

BAD (too vague):  "Add a pause feature"
BAD (has code):   "Add: if (isPaused) clearInterval(gameLoop);"
GOOD (detailed):  "Add a pause feature: when the user presses Space, the game
      loop timer should be stopped and a semi-transparent overlay should appear
      over the canvas with the text 'PAUSED' centered. Pressing Space again
      should restart the timer from where it stopped and hide the overlay.
      The pause state should be tracked with a boolean. While paused, keyboard
      input for direction changes should be ignored."

⚠️ FOLLOW THE USER'S INSTRUCTIONS EXACTLY:
   - If the user says HOW to do something, do it THAT WAY. Do not choose
     a different approach because you think it's better.
   - If the user gives specific details (colors, sizes, names, behaviors),
     use those EXACT details. Do not substitute your own preferences.
   - If the user says "use X library", use X. If they say "make it blue",
     make it blue. If they say "put it in file Y", put it in file Y.
   - The user's instructions are not suggestions. They are requirements.

⚠️ ANTI-DRIFT: Before writing each step, ask yourself:
   "Does this step directly serve the USER'S TASK, or am I adding something
   they didn't ask for?" Only include what the task requires. Do NOT add
   improvements, refactors, or "nice to haves" unless the user asked for them.

⚠️ COMPLETENESS: Re-read the TASK above. Count every separate thing it asks for.
   If the task says "do X and Y", your plan MUST cover both X and Y.
   If the task says "do A, B, and C", your plan MUST have steps for A, B, AND C.
   Before finishing, go back to the TASK and check off each request:
   - Did I plan for this? If no, add it now.
   A plan that only covers half the task is a BAD plan. Cover EVERYTHING.
══════════════════════════════════════════════════════════════════════

0. INFER INTENT: Restate what the user wants in your own words.
   If ambiguous, use the conversation context to clarify.
   State EXACTLY what you will and will NOT change.

   IMPORTANT — distinguish CONTEXT from INSTRUCTIONS in the user's message:
   The user's prompt often contains BOTH:
     - CONTEXT: describing the CURRENT state of the code ("right now the sidebar
       is 200px wide", "currently it uses a flat list", "the button is red")
       → This is NOT a request. This is background info explaining what EXISTS.
     - INSTRUCTIONS: what they want you to CHANGE ("make the sidebar collapsible",
       "switch to a tree structure", "change it to blue")
       → THIS is what you need to plan.
   
   Do NOT treat descriptions of current behavior as things to implement.
   "The app has a login page" = context (it already exists).
   "Add a login page" = instruction (build it).
   "The colors are too bright" = context (problem statement).
   "Make the colors more muted" = instruction (what to do about it).

   LIST every separate INSTRUCTION the task asks for — numbered:
     1. [first thing the user wants CHANGED]
     2. [second thing]
     3. ...
   If the user specified HOW to do any of these, note the approach they want.
   Your plan must address ALL of them, using the approach the user specified.

## PLAN SUMMARY
[One paragraph: what we're doing, why, and the high-level approach]

## DETAILED STEPS
For each step:
1. [File: path] [Section: name]
   Current behavior: ...
   New behavior: ...
   Logic: ... (in English, no code)

2. ...

## FILES TO MODIFY
- path/to/file — what changes and why

## FILES TO CREATE (if any)
- path/to/new_file — what it contains and its purpose

## EDGE CASES AND ERROR HANDLING
- What could go wrong and how to handle each case

## TEST CRITERIA (how to verify it works)
- Specific testable behaviors, not vague "it should work"
"""

IMPL_COT_EXISTING = """
══════════════════════════════════════════════════════════════════════
YOUR CHAIN OF THOUGHT — EXPAND LEVEL BY LEVEL
══════════════════════════════════════════════════════════════════════

You MUST think by EXPANDING, level by level. Never jump to writing code.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 1 — LOCATE IN THE PLAN                                  │
└─────────────────────────────────────────────────────────────────┘

Read the plan above. Which section(s) apply to THIS file?
Write your thinking:
  "The plan says this file needs: [summarize relevant plan steps].
   This file currently does: [summarize from the file content above].
   The changes involve: [what needs to change at a high level]."

⚠️ ANTI-DRIFT: Implement ONLY what the plan says. Do NOT:
  - Add features or improvements not in the plan
  - Refactor code that the plan doesn't mention
  - Fix unrelated bugs you happen to notice
  - Change coding style or formatting of untouched code
  If it's not in the plan, don't touch it.

┌─────────────────────────────────────────────────────────────────┐
│  STEP 1 — UNDERSTAND YOUR JOB                                   │
└─────────────────────────────────────────────────────────────────┘

Read the PLAN above. Which steps apply to THIS file?
Write: "This file needs: [list the changes from the plan]"

If you need to see code from OTHER files to implement correctly
(e.g. a function signature you need to match), look it up now:
  [CODE: path/file] or [REFS: name]
  STOP

Only look up what you actually need. If the plan already tells you
everything, skip straight to Step 2.

┌─────────────────────────────────────────────────────────────────┐
│  STEP 2 — WRITE THE CODE                                        │
└─────────────────────────────────────────────────────────────────┘

Write the code edits using the format below.

For EDIT blocks: copy-paste the SEARCH text from the CURRENT FILE CONTENT
above. Do NOT retype from memory. If even one character is wrong, the
edit WILL FAIL.

┌─────────────────────────────────────────────────────────────────┐
│  STEP 3 — CHECK YOUR WORK (MANDATORY — syntax errors = extra  │
│           round trip to fix them, so get it right now)          │
└─────────────────────────────────────────────────────────────────┘

Your code will be SYNTAX-CHECKED after you submit. If it has syntax
errors, you'll be called AGAIN to fix them — wasting time. Avoid that
by checking carefully now.

SYNTAX CHECKLIST — go through EVERY item:
  □ Brackets/parens/braces: count every ( and ) — do they match?
    Count every { and } — do they match? Every [ and ]?
  □ Strings: every opening quote has a matching close? No mixing
    single and double quotes mid-string? F-strings have valid
    expressions inside {}?
  □ Python: every if/for/while/def/class ends with a colon?
    Indentation is consistent (all spaces or all tabs, never mixed)?
    No stray dedents?
  □ JS/TS: every statement ends with ; (if project uses them)?
    Arrow functions have => not just >? Template literals use
    backticks, not regular quotes?
  □ Imports: every name you use is imported or defined? No typos
    in import paths?
  □ SEARCH text matches the original file EXACTLY?
  □ Did you implement every plan step for this file?
  □ Go back to the TASK — did you cover everything it asks for?
  □ Did you change any function name, parameter, or return type?
    If yes: use [LSP: name] STOP — check dependencies won't break.

If you find ANY flaw, write corrected EDIT blocks now.

══════════════════════════════════════════════════════════════════════
"""

IMPL_COT_NEW = """
══════════════════════════════════════════════════════════════════════
YOUR CHAIN OF THOUGHT — EXPAND LEVEL BY LEVEL
══════════════════════════════════════════════════════════════════════

This is a NEW file — there is no existing code to look up. Do NOT use
[DETAIL:], [CODE:], or [SEARCH:] tags.

You MUST still think by EXPANDING, level by level. Never jump to code.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 1 — LOCATE IN THE PLAN                                  │
└─────────────────────────────────────────────────────────────────┘

Read the plan above. Which section(s) apply to THIS file?
Write your thinking:
  "The plan says this file should: [summarize relevant plan steps].
   Its role in the project is: [how it fits with other files].
   Other files will depend on it for: [what it exports/provides]."

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 2 — EXPAND PLAN INTO DETAILED DESIGN                    │
└─────────────────────────────────────────────────────────────────┘

Now expand the plan section into a precise design for this file.
Write out:
  - What functions/classes will this file contain?
  - For EACH function: what are the inputs, outputs, and logic?
  - What data structures and state does it manage?
  - What does it export? What will other files import from it?
  - What edge cases should each function handle?
  - What is the order of operations / control flow?

Do NOT write code yet. Just the detailed design in English.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 3 — EXPAND DESIGN INTO CODE                             │
└─────────────────────────────────────────────────────────────────┘

Now — and ONLY now — write the actual code using the format below.
Translate your Level 2 design function by function into real code.

You can use [WEBSEARCH: query] if you need to look up an API or library.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 4 — FIND FLAWS (syntax errors = extra fix round trip)   │
└─────────────────────────────────────────────────────────────────┘

STOP. Your code will be SYNTAX-CHECKED after you submit. If it has
errors, you'll be called again to fix them — wasting a round trip.
Check carefully now to avoid that.

SYNTAX CHECKLIST — go through EVERY item:
  □ Brackets: count every ( and ) — do they match? Every { and }?
    Every [ and ]? Mismatched brackets are the #1 syntax error.
  □ Strings: every opening quote (' or " or ` or triple-quotes) has a matching
    close? No f-string with unescaped { inside? No mixing quote types?
  □ Python: every if/for/while/def/class line ends with ":"?
    Indentation is consistent (all spaces, never tabs mixed in)?
    No accidental dedent in the middle of a function?
  □ JS/TS: arrow functions use "=>" not ">"? Template literals use
    backticks not quotes? Every switch case has break or return?
  □ HTML: every opening tag has a closing tag (except void elements)?
    No unclosed <div>, <span>, <p>?
  □ Imports: every name you reference is imported or defined above?
    Module paths are correct (relative vs absolute)?

Then also check:
  - Logic errors: off-by-one, wrong conditions, missing returns?
  - Missing imports or dependencies?
  - Plan compliance: does the file fulfill its role in the plan?
  - Every function from Level 2 is implemented?
  - Edge cases: what inputs could break this code?
  - Consistency: will other files be able to use this as expected?

Write out every flaw you find. Be harsh — it's better to catch it now.

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 5 — FIX THE FLAWS                                       │
└─────────────────────────────────────────────────────────────────┘

For EACH flaw you found in Level 4, fix it now.
Output corrected code that replaces the broken parts.

If you found no flaws, confirm: "No flaws found — code is correct."

┌─────────────────────────────────────────────────────────────────┐
│  LEVEL 6 — VERIFY AGAINST THE TASK                              │
└─────────────────────────────────────────────────────────────────┘

FINAL CHECK. Go back to the TASK and the PLAN at the top of this prompt.
Read them again. For each thing the task asks for, check:
  - Did I implement it? Where in my code?
  - If the task asked for 3 things, do I see all 3?
  - Did I miss anything? If yes, add the missing code now.

══════════════════════════════════════════════════════════════════════
"""

IMPLEMENT_PROMPT = SYSTEM_KNOWLEDGE + """

══════════════════════════════════════════════════════════════════════
YOUR ROLE: CODER
══════════════════════════════════════════════════════════════════════
You are a developer. Your ONLY job is to write code that implements the plan.

WHY YOU EXIST: Multiple AIs wrote plans, debated them, and merged the best
ideas into the final plan you see below. A SEPARATE AI will review your code
after you finish — it will check for bugs, missing steps, and improvements.
Write clean, correct code the first time.

⚠️ SYNTAX CHECKER: Your output is AUTOMATICALLY SYNTAX-CHECKED.
If it has syntax errors (mismatched brackets, unclosed strings, bad
indentation, missing colons, etc.), you will be called AGAIN to fix them.
Getting it right the first time saves a round trip — triple-check your syntax.

YOU DO:
  - Read the plan and implement it EXACTLY as specified
  - Write correct, working code with proper syntax
  - Expand level by level: plan -> detailed design -> code
  - Self-verify your code before finishing

YOU DO NOT:
  - Change the plan — if you disagree, implement it anyway
  - Skip plan steps — every step must be implemented
  - Add features not in the plan
  - Guess at function names — look them up

TOOLS — escalate only if needed:
  1. [REFS: name]        — find where a name is defined, imported, used
  2. [LSP: name]         — semantic dependencies, types (use before changing interfaces)
  3. [DETAIL: section]   — code map for a feature
     [PURPOSE: category] — all code for a purpose
  4. [CODE: path/file]   — read actual source code
     [SEARCH: pattern]   — ripgrep search
     [WEBSEARCH: query]  — web search for API docs
Write tags, then STOP.
══════════════════════════════════════════════════════════════════════

TASK: {task}

PLAN:
{plan}

CURRENT FILE CONTENT (the FULL file — you can see everything):
{file_content}

PROJECT OVERVIEW:
{context}

{cot_instructions}

OUTPUT FORMAT — depends on whether you're EDITING or CREATING:

For EDITING an existing file — output ONLY the changes, not the whole file:
=== EDIT: path/to/file.py ===
[SEARCH]
the exact lines from the original file that you want to replace
(copy them EXACTLY — whitespace matters)
[/SEARCH]
[REPLACE]
the new lines that should go there instead
[/REPLACE]

You can have MULTIPLE search/replace blocks per file:
=== EDIT: path/to/file.py ===
[SEARCH]
first section to change
[/SEARCH]
[REPLACE]
replacement for first section
[/REPLACE]

[SEARCH]
second section to change
[/SEARCH]
[REPLACE]
replacement for second section
[/REPLACE]

To ADD new code (insert after a line), use SEARCH with the line BEFORE where you want to insert:
[SEARCH]
line_that_exists_before_insertion_point
[/SEARCH]
[REPLACE]
line_that_exists_before_insertion_point
new_code_here
[/REPLACE]

To DELETE code, use an EMPTY replace block — the matched lines will be removed entirely:
[SEARCH]
the lines you want to delete
[/SEARCH]
[REPLACE]
[/REPLACE]

This is how you REMOVE old code that is no longer needed. When modifying a feature,
you often need to DELETE the old implementation AND add a new one. Use separate
SEARCH/REPLACE blocks: one with empty REPLACE to delete, one to add the new code.

For CREATING a new file — write the ENTIRE file, every single line, fully complete and working:
=== FILE: path/to/newfile.ext ===
```language
THE COMPLETE FILE — every line, every function, every import.
Do NOT skip anything. Do NOT write "// ... rest of code".
Do NOT abbreviate. Write the FULL working file from top to bottom.
```

CRITICAL RULES:
- EXISTING files: NEVER rewrite the whole file. Only output EDIT blocks with the parts that CHANGE.
- NEW files: ALWAYS write the COMPLETE file. Every line. No shortcuts. No placeholders.
- The SEARCH text must match the original file EXACTLY — copy-paste from the
  CURRENT FILE CONTENT section above. Do NOT retype it from memory. If even one
  character is wrong, the edit WILL FAIL and be skipped.
- Do NOT use git merge conflict markers (<<<<<<, =======, >>>>>>). Use [SEARCH] and [REPLACE] tags.
- Output ONLY the edit/file blocks. No explanation, no commentary. JUST the edits and files.
- Output ONLY ONE FILE per response. If you need to create/edit multiple files,
  you will be called once per file. Focus on the file specified below.

SAFETY RULE:
Before changing ANYTHING that other files depend on (function names, parameters,
return types, exported names), use [LSP: name] to check dependencies.
LSP shows semantic connections — what depends on this function, what types
flow through it. If callers exist, either keep the interface the same or
note in your edit that callers need updating too.
Do NOT blindly rename, remove, or change function signatures.

YOU ARE IMPLEMENTING THIS FILE: {target_file}
"""

SUMMARY_PROMPT = """You just implemented code changes. Summarize what you did for the user.

TASK: {task}

FILES CHANGED:
{files_changed}

DIFF:
{diff}

Write a clear, concise summary in plain English:
1. What files were created or modified
2. What each change does
3. Any important things the user should know (new dependencies, config changes, etc.)

Keep it short — the user wants to understand what changed, not read the code again.
Do NOT include any code in your summary.
"""

MAP_UPDATE_PROMPT = """You just implemented code changes. You need to update the project's code maps.

DO NOT rewrite the maps. DO NOT restate them in your thinking.
Output ONLY edit blocks for the specific parts that actually changed.

TASK COMPLETED: {task}

FILES CHANGED:
{files_changed}

DIFF OF CHANGES:
{diff}

CURRENT GENERAL MAP:
{general_map}

CURRENT DETAILED MAP:
{detailed_map}

YOUR JOB: Output edit blocks for each map that needs updating.
If a map doesn't need changes, write "GENERAL: no changes" or "DETAILED: no changes".

EDIT FORMAT — wrap edits with map headers:

=== GENERAL MAP EDITS ===
[SEARCH]
exact text from current general map
[/SEARCH]
[REPLACE]
new text (or empty to delete)
[/REPLACE]

[ADD_SECTION]
## New Feature Name
description of new feature
[/ADD_SECTION]

=== DETAILED MAP EDITS ===
[SEARCH]
exact text from current detailed map
[/SEARCH]
[REPLACE]
updated text
[/REPLACE]

[ADD_SECTION]
=== SECTION: New Feature ===
### file.py — new_function(params)
  Purpose: ...
[/ADD_SECTION]

RULES:
- Copy SEARCH text EXACTLY from the current map (even one character off = edit ignored)
- Only edit what the diff actually changed
- Empty REPLACE deletes the matched text
- ADD_SECTION appends to the end of that map
- Keep your analysis SHORT. Your output is edit blocks, not prose.
"""


def _apply_map_edits(original_map: str, response_text: str) -> str:
    """Parse [SEARCH]/[REPLACE] and [ADD_SECTION] blocks from response and
    apply them to the original map. Empty [REPLACE] deletes."""
    result = original_map

    # Find SEARCH/REPLACE pairs
    edit_pattern = re.compile(
        r'\[SEARCH\](.*?)\[/SEARCH\]\s*\[REPLACE\](.*?)\[/REPLACE\]',
        re.DOTALL,
    )
    for match in edit_pattern.finditer(response_text):
        find_text = match.group(1).strip()
        replace_text = match.group(2).strip()
        if not find_text:
            continue
        if find_text in result:
            result = result.replace(find_text, replace_text, 1)
        else:
            # Fuzzy: whitespace-normalized line match
            find_lines = [l.strip() for l in find_text.split('\n')]
            result_lines = result.split('\n')
            for i in range(len(result_lines) - len(find_lines) + 1):
                window = [result_lines[i + j].strip() for j in range(len(find_lines))]
                if window == find_lines:
                    if replace_text:
                        result_lines[i:i + len(find_lines)] = replace_text.split('\n')
                    else:
                        del result_lines[i:i + len(find_lines)]
                    result = '\n'.join(result_lines)
                    break

    # Find ADD_SECTION blocks — append to end
    add_pattern = re.compile(r'\[ADD_SECTION\](.*?)\[/ADD_SECTION\]', re.DOTALL)
    for match in add_pattern.finditer(response_text):
        addition = match.group(1).strip()
        if addition:
            result += "\n\n" + addition

    return result


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _format_research_cache(research_cache: dict | None, max_chars: int = 30000) -> str:
    """Format the shared research cache into a readable section for prompt injection.

    This lets downstream AIs (coders, reviewers) see everything that upstream
    AIs (planners) already looked up, so they don't need to re-search.
    Results are deduplicated by key — identical lookups only appear once.
    """
    if not research_cache:
        return ""

    parts = []
    total = 0
    for key, value in research_cache.items():
        value = value.strip()
        if not value:
            continue
        # key format is "TAG_TYPE:query" e.g. "REFS:call_with_tools"
        entry = f"\n{value}"
        if total + len(entry) > max_chars:
            parts.append(f"\n... ({len(research_cache) - len(parts)} more cached lookups truncated)")
            break
        parts.append(entry)
        total += len(entry)

    if not parts:
        return ""

    return (
        "\n\n══════════════════════════════════════════════════════════════\n"
        "PRE-LOADED RESEARCH (from earlier pipeline stages — do NOT re-search these):\n"
        "The planning AIs already looked these up. Use this data directly.\n"
        "If you need something NOT listed here, you can still use tool tags.\n"
        "══════════════════════════════════════════════════════════════\n"
        + "\n".join(parts)
        + "\n══════════════════════════════════════════════════════════════\n"
    )


def _check_syntax(filepath: str, content: str) -> tuple[bool, str]:
    """Run a syntax check on code content based on file extension.

    Returns (passed: bool, error_message: str).
    If no checker is available for the file type, returns (True, "").
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".py":
        try:
            compile(content, filepath, "exec")
            return True, ""
        except SyntaxError as e:
            lineno = e.lineno or "?"
            offset = e.offset or ""
            msg = e.msg or "unknown syntax error"
            # Extract the offending line for context
            lines = content.split("\n")
            context_lines = []
            if isinstance(e.lineno, int) and e.lineno > 0:
                start = max(0, e.lineno - 3)
                end = min(len(lines), e.lineno + 2)
                for i in range(start, end):
                    marker = ">>>" if i == e.lineno - 1 else "   "
                    context_lines.append(f"  {marker} {i + 1}: {lines[i]}")
            context_str = "\n".join(context_lines)
            return False, (
                f"Python SyntaxError at line {lineno}, col {offset}: {msg}\n"
                f"{context_str}"
            )

    elif ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
        # Use Node.js --check for JS/TS syntax validation
        import tempfile
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=ext, delete=False, encoding="utf-8"
            ) as f:
                f.write(content)
                tmp_path = f.name

            # Try node --check (works for .js/.mjs/.cjs)
            if ext in (".js", ".mjs", ".cjs"):
                result = subprocess.run(
                    ["node", "--check", tmp_path],
                    capture_output=True, text=True, timeout=10,
                )
                if result.returncode != 0:
                    err = result.stderr.strip()
                    # Clean up the temp path from the error message
                    err = err.replace(tmp_path, filepath)
                    return False, f"JavaScript syntax error:\n{err}"

            return True, ""
        except FileNotFoundError:
            # node not installed — skip
            return True, ""
        except subprocess.TimeoutExpired:
            return True, ""
        except Exception:
            return True, ""
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    elif ext in (".json",):
        import json
        try:
            json.loads(content)
            return True, ""
        except json.JSONDecodeError as e:
            return False, f"JSON syntax error at line {e.lineno}, col {e.colno}: {e.msg}"

    elif ext in (".html", ".htm"):
        # Basic tag balance check — catches unclosed tags
        open_tags = re.findall(r'<([a-zA-Z][a-zA-Z0-9]*)\b[^/]*?(?<!/)>', content)
        close_tags = re.findall(r'</([a-zA-Z][a-zA-Z0-9]*)\s*>', content)
        void_elements = {
            "area", "base", "br", "col", "embed", "hr", "img", "input",
            "link", "meta", "param", "source", "track", "wbr",
        }
        open_tags = [t.lower() for t in open_tags if t.lower() not in void_elements]
        close_tags = [t.lower() for t in close_tags]
        if len(open_tags) != len(close_tags):
            diff = len(open_tags) - len(close_tags)
            return False, (
                f"HTML tag mismatch: {len(open_tags)} opening vs {len(close_tags)} closing tags "
                f"({abs(diff)} {'unclosed' if diff > 0 else 'extra closing'} tag(s))"
            )
        return True, ""

    elif ext in (".css",):
        # Basic brace balance check
        opens = content.count("{")
        closes = content.count("}")
        if opens != closes:
            diff = opens - closes
            return False, (
                f"CSS brace mismatch: {opens} opening vs {closes} closing braces "
                f"({abs(diff)} {'unclosed' if diff > 0 else 'extra'} brace(s))"
            )
        return True, ""

    # No checker for this file type
    return True, ""


SYNTAX_FIX_PROMPT = SYSTEM_KNOWLEDGE + """

══════════════════════════════════════════════════════════════════════
YOUR ROLE: SYNTAX FIXER
══════════════════════════════════════════════════════════════════════
Another AI just wrote code for this file, but it has a SYNTAX ERROR.
Your ONLY job is to fix the syntax error. Do NOT change any logic,
do NOT add features, do NOT refactor. JUST fix the syntax.

FILE: {target_file}

SYNTAX ERROR:
{syntax_error}

THE BROKEN CODE (full file):
{broken_code}

FIX the syntax error using EDIT blocks. Change the MINIMUM amount of
code needed to make the syntax valid.

Common fixes:
  - Add a missing closing bracket ) ] or brace }}
  - Add a missing colon : after if/for/while/def/class (Python)
  - Fix indentation (Python)
  - Close an unclosed string quote
  - Add a missing semicolon (JS/TS)
  - Fix mismatched HTML tags

OUTPUT FORMAT:
=== EDIT: {target_file} ===
[SEARCH]
the exact broken line(s) — copy from THE BROKEN CODE above
[/SEARCH]
[REPLACE]
the fixed line(s)
[/REPLACE]

Output ONLY the edit blocks. No explanation.
"""


async def _syntax_fix_loop(
    target_file: str,
    content: str,
    project_root: str,
    max_fix_attempts: int = 2,
    detailed_map: str = "",
    purpose_map: str = "",
    research_cache: dict | None = None,
) -> str | None:
    """Check syntax and auto-fix errors by sending broken code back to the AI.

    Instead of throwing away the code and starting over, this sends the broken
    code + the exact error to the AI and asks it to fix JUST the syntax.
    Up to max_fix_attempts rounds of fix. Returns the fixed code, or None
    if it couldn't be fixed.
    """
    passed, syntax_err = _check_syntax(target_file, content)
    if passed:
        return content  # already clean

    for fix_round in range(1, max_fix_attempts + 1):
        warn(f"  Syntax error in {target_file}: {syntax_err.split(chr(10))[0]}")
        status(f"  Syntax fix round {fix_round}/{max_fix_attempts}...")

        fix_prompt = SYNTAX_FIX_PROMPT.format(
            target_file=target_file,
            syntax_error=syntax_err,
            broken_code=content,
        )

        fix_result = await _call_with_tools(
            IMPLEMENT_MODEL, fix_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            max_tokens=8192,
            log_label=f"fixing syntax: {target_file} (round {fix_round})",
        )

        extracted = _extract_code_blocks(fix_result["answer"])

        # Apply the fix edits to the broken code
        if extracted["edits"]:
            edit_pairs = []
            for fp, pairs in extracted["edits"].items():
                edit_pairs.extend(pairs)
            if edit_pairs:
                content = _apply_edits(content, edit_pairs)
        elif extracted["new_files"]:
            # AI rewrote the whole file — accept it
            for fname, fcontent in extracted["new_files"].items():
                content = fcontent
                break
        else:
            warn(f"  Syntax fixer produced no edits — giving up")
            return None

        # Check again
        passed, syntax_err = _check_syntax(target_file, content)
        if passed:
            success(f"  Syntax fixed in {target_file} (round {fix_round})")
            return content

    # Still broken after all fix attempts
    warn(f"  Could not fix syntax in {target_file} after {max_fix_attempts} rounds")
    return None


async def _call(model: str, prompt: str, max_tokens: int = 16384, log_label: str = "") -> dict:
    result = await call_with_retry(model, prompt, max_tokens=max_tokens, log_label=log_label)
    return {"model": model, "answer": result}


# _call_with_tools imported from core.tool_call (shared with chat + research)


def _extract_code_blocks(response: str) -> dict:
    """
    Extract edits and new files from AI response.
    Handles:
      - [SEARCH]...[/SEARCH] [REPLACE]...[/REPLACE] tags (primary)
      - <<<< FIND ... >>>> <<<< REPLACE ... >>>> (fallback)
      - <<<< FIND ... ======= ... >>>> (git-style fallback)
      - === FILE: path === ```code``` (new files)
      - Plain ```code``` blocks (last resort)
    Returns {
        "edits": {filepath: [(search, replace), ...]},
        "new_files": {filepath: content},
    }
    """
    result = {"edits": {}, "new_files": {}}

    # ── Extract EDIT blocks ──────────────────────────────────────────────
    edit_pattern = re.compile(
        r'===\s*EDIT:\s*(.+?)\s*===\s*(.*?)(?====\s*(?:EDIT|FILE):|$)',
        re.DOTALL
    )
    for edit_match in edit_pattern.finditer(response):
        filepath = edit_match.group(1).strip()
        edit_body = edit_match.group(2)

        pairs = []

        # Format 1: [SEARCH]...[/SEARCH] [REPLACE]...[/REPLACE]
        xml_pairs = re.findall(
            r'\[SEARCH\]\s*\n?(.*?)\s*\[/SEARCH\]\s*\n?\s*\[REPLACE\]\s*\n?(.*?)\s*\[/REPLACE\]',
            edit_body, re.DOTALL
        )
        if xml_pairs:
            pairs.extend(xml_pairs)

        # Format 2: <<<< FIND ... >>>> <<<< REPLACE ... >>>>
        if not pairs:
            old_pairs = re.findall(
                r'<<<<\s*FIND\s*\n?(.*?)\s*>>>>\s*\n?\s*<<<<\s*REPLACE\s*\n?(.*?)\s*>>>>',
                edit_body, re.DOTALL
            )
            if old_pairs:
                pairs.extend(old_pairs)

        # Format 3: <<<< FIND ... ======= ... >>>> (git merge style)
        if not pairs:
            git_pairs = re.findall(
                r'<<<<\s*FIND\s*\n?(.*?)\s*=======\s*\n?(.*?)\s*>>>>',
                edit_body, re.DOTALL
            )
            if git_pairs:
                pairs.extend(git_pairs)

        # Format 4: <<<<<<< ... ======= ... >>>>>>> (actual git markers)
        if not pairs:
            git2_pairs = re.findall(
                r'<{4,7}\s*\w*\s*\n?(.*?)\s*={4,7}\s*\n?(.*?)\s*>{4,7}',
                edit_body, re.DOTALL
            )
            if git2_pairs:
                pairs.extend(git2_pairs)

        if pairs:
            result["edits"].setdefault(filepath, []).extend(pairs)

    # ── Extract FILE blocks (new files) ──────────────────────────────────
    file_pattern = re.compile(
        r'===\s*FILE:\s*(.+?)\s*===\s*```[^\n]*\n(.*?)```',
        re.DOTALL
    )
    for file_match in file_pattern.finditer(response):
        filepath = file_match.group(1).strip()
        content = file_match.group(2).strip()
        result["new_files"][filepath] = content

    # ── Fallback: plain code blocks ──────────────────────────────────────
    if not result["edits"] and not result["new_files"]:
        all_blocks = re.findall(r'```[^\n]*\n(.*?)```', response, re.DOTALL)
        if all_blocks:
            longest = max(all_blocks, key=len)
            result["new_files"]["main"] = longest.strip()

    return result


def _apply_edits(original: str, edits: list[tuple[str, str]]) -> str:
    """Apply FIND/REPLACE edits to original file content.
    Empty replace = deletion (removes matched lines entirely).
    
    Three matching strategies:
      1. Exact match (fastest)
      2. Whitespace-normalized sliding window
      3. Fuzzy match with SequenceMatcher (catches AI memory errors)
    """
    import difflib

    result = original
    for find_text, replace_text in edits:
        find_clean = find_text.strip()
        replace_clean = replace_text.strip()

        # ── Strategy 1: Exact match ──────────────────────────────────
        if find_clean in result:
            if not replace_clean:
                # Deletion: remove the lines and clean up blank lines
                lines = result.split('\n')
                find_lines = find_clean.split('\n')
                find_stripped = [l.strip() for l in find_lines]
                for i in range(len(lines) - len(find_lines) + 1):
                    window = [lines[i + j].strip() for j in range(len(find_lines))]
                    if window == find_stripped:
                        del lines[i:i + len(find_lines)]
                        result = '\n'.join(lines)
                        break
                else:
                    result = result.replace(find_clean, '', 1)
            else:
                result = result.replace(find_clean, replace_clean, 1)
            continue

        # ── Strategy 2: Whitespace-normalized sliding window ─────────
        find_lines = [l.strip() for l in find_clean.split('\n')]
        result_lines = result.split('\n')

        found = False
        for i in range(len(result_lines) - len(find_lines) + 1):
            window = [result_lines[i + j].strip() for j in range(len(find_lines))]
            if window == find_lines:
                result_lines, result, found = _do_replace(
                    result_lines, i, len(find_lines), replace_clean
                )
                break

        if found:
            continue

        # ── Strategy 3: Fuzzy match ──────────────────────────────────
        # The AI wrote the SEARCH block from memory — find the closest
        # matching block in the file using SequenceMatcher
        find_joined = "\n".join(find_lines)  # whitespace-normalized
        best_score = 0.0
        best_idx = -1
        best_length = len(find_lines)
        window_sizes = [len(find_lines), len(find_lines) - 1, len(find_lines) + 1]

        for wsize in window_sizes:
            if wsize < 1 or wsize > len(result_lines):
                continue
            for i in range(len(result_lines) - wsize + 1):
                window = [result_lines[i + j].strip() for j in range(wsize)]
                window_joined = "\n".join(window)
                score = difflib.SequenceMatcher(
                    None, find_joined, window_joined
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_idx = i
                    best_length = wsize

        if best_score >= 0.6 and best_idx >= 0:
            from core.cli import warn as _warn, success as _success
            _success(f"Fuzzy matched FIND block ({best_score:.0%} similarity)")
            result_lines_current = result.split('\n')
            result_lines_current, result, _ = _do_replace(
                result_lines_current, best_idx, best_length, replace_clean
            )
        else:
            from core.cli import warn, error
            preview = find_clean[:80].replace('\n', '\\n')
            error(f"FIND block not matched — SKIPPING edit (not appending)")
            warn(f"  Tried to find: {preview}...")
            if best_score > 0:
                warn(f"  Best fuzzy match was {best_score:.0%} (need 60%+)")

    return result


def _do_replace(
    result_lines: list[str], start: int, length: int, replace_clean: str
) -> tuple[list[str], str, bool]:
    """Apply a replacement at a matched position, preserving indentation.
    If replace_clean is empty, deletes the matched lines entirely."""
    # Empty replace = deletion
    if not replace_clean.strip():
        result_lines[start:start + length] = []
        return result_lines, '\n'.join(result_lines), True

    # Get the indentation of the first matched line
    indent = result_lines[start][:len(result_lines[start]) - len(result_lines[start].lstrip())]
    # Preserve RELATIVE indentation from the replacement
    replace_lines = replace_clean.split('\n')
    # Find the base indent of the replacement (first non-empty line)
    base_replace_indent = 0
    for rl in replace_lines:
        if rl.strip():
            base_replace_indent = len(rl) - len(rl.lstrip())
            break
    new_lines = []
    for line in replace_lines:
        if line.strip():
            line_indent = len(line) - len(line.lstrip())
            relative = max(0, line_indent - base_replace_indent)
            new_lines.append(indent + ' ' * relative + line.strip())
        else:
            new_lines.append('')
    result_lines[start:start + length] = new_lines
    return result_lines, '\n'.join(result_lines), True


async def _run_searches_and_augment(ai_response: str, project_root: str, context: str) -> str:
    """Run any [SEARCH: pattern] tags found in AI response, append results to context."""
    extra = await run_on_demand_searches(ai_response, project_root)
    if extra:
        context += f"\n\n=== ON-DEMAND SEARCH RESULTS ===\n{extra}"
    return context


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — UNDERSTAND
# ═══════════════════════════════════════════════════════════════════════════════

async def phase_understand(task: str, project_root: str) -> dict:
    """
    3 AIs search codebase in parallel. Union ALL findings.
    For empty/new projects: skips searching, returns minimal context.
    Returns: {context: str, files: list[str], classification: str}
    """
    step("═══ Phase 1: UNDERSTAND ═══")

    # Scan project structure
    project_structure = scan_project(project_root)
    # Extract file count from last line: "(N files)"
    file_count_match = re.search(r'\((\d+) files?\)', project_structure)
    file_count = int(file_count_match.group(1)) if file_count_match else 0

    status(f"Project scanned: {file_count} files")

    # ── Empty/new project: skip heavy searching ──────────────────────────
    if file_count == 0:
        status("Empty project — creating from scratch")
        context = (
            f"PROJECT: {project_root}\n"
            f"STATUS: Empty directory — this is a NEW project.\n"
            f"TASK: {task}\n\n"
            f"There are no existing files. Create all files from scratch."
        )
        return {
            "context": context,
            "files": [],
            "project_structure": project_structure,
        }

    # ── Existing project: 3 AIs search in parallel ────────────────────
    step("3 AIs searching codebase...")
    prompt = UNDERSTAND_PROMPT.format(
        task=task,
        project_structure=project_structure[:8000],
    )

    results = list(await asyncio.gather(
        *[_call_with_tools(m, prompt, project_root, log_label="understanding codebase") for m in UNDERSTAND_MODELS],
        return_exceptions=True,
    ))
    results = [r for r in results if isinstance(r, dict) and r.get("answer")]

    if not results:
        raise RuntimeError("All 3 AIs failed in Phase 1")

    # Union ALL findings — if ANY AI says it's relevant, include it
    all_files = set()
    all_context_parts = []

    for r in results:
        ai_name = r["model"].split("/")[-1]
        all_context_parts.append(f"\n=== {ai_name}'s analysis ===\n{r['answer']}")

        # Extract file paths mentioned
        for line in r["answer"].split("\n"):
            # Match common path patterns
            path_match = re.findall(r'[\w./\-]+\.(?:py|js|ts|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md)', line)
            all_files.update(path_match)

    # Read all mentioned files into context
    for fpath in sorted(all_files):
        full_path = os.path.join(project_root, fpath)
        content = read_file(full_path)
        if content and not content.startswith("["):
            all_context_parts.append(f"\n══ {fpath} ══\n{content}")

    full_context = "\n".join(all_context_parts)
    status(f"Phase 1: {len(results)} AIs, {len(all_files)} files, {len(full_context)} chars context")
    success("Phase 1 complete — code understood")

    return {
        "context": full_context,
        "files": sorted(all_files),
        "project_structure": project_structure,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — PLAN (multi-model generate → single AI merge)
# ═══════════════════════════════════════════════════════════════════════════════

async def phase_plan(task: str, context: str, complexity: int, project_root: str,
                     plan_feedback: str = "", detailed_map: str = "",
                     purpose_map: str = "",
                     is_new_project: bool = False) -> tuple[str, dict]:
    """
    Planning:
      Standard (complexity < 7):
        Layer 1: 4 AIs race, first 3 win, last is cancelled
        Layer 2: GLM-5 merges the 3 winning plans into final plan
      Deep (complexity >= 7):
        Layer 1: 4 AIs race, first 3 win, last is cancelled
        Layer 2: 4 AIs each read the 3 plans, find flaws/strengths,
                 write their own improved plan (parallel)
        Layer 3: GLM-5 reads all 4 improved plans, writes final plan

    Research from all planning AIs is accumulated in a shared cache and
    returned so implement + review can reuse it without re-running lookups.

    Returns (final_plan, research_cache).
    """
    extended = complexity >= 7
    mode_label = "EXTENDED 3-layer" if extended else "STANDARD"
    step(f"=== Phase 2: PLAN [{mode_label}] ===")

    PLAN_MODELS = [
        "nvidia/deepseek-v3.2",
        "nvidia/qwen-3.5",
        "nvidia/minimax-m2.5",
        "nvidia/nemotron-super",
    ]

    cot = PLAN_COT_NEW if is_new_project else PLAN_COT_EXISTING
    plan_prompt = PLAN_PROMPT.format(
        task=task,
        context=context[:30000],
        cot_instructions=cot,
    )

    if plan_feedback:
        plan_prompt += (
            f"\n\nPREVIOUS PLAN WAS REJECTED:\n{plan_feedback}\n"
            f"Fix the issues above. Do NOT repeat the same mistakes."
        )

    # Shared research cache — accumulates lookups across all AIs in this workflow
    research_cache: dict[str, str] = {}

    # == Layer 1: 4 AIs RACE — keep first 3, kill the last ==
    step(f"Layer 1: {len(PLAN_MODELS)} models racing (first 3 win)...")

    plan_tasks = [
        asyncio.create_task(_call_with_tools(
            m, plan_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label=f"planning (Layer 1)",
        ))
        for m in PLAN_MODELS
    ]

    # Wait for 3 to complete, cancel the rest
    done, pending = await asyncio.wait(
        plan_tasks, return_when=asyncio.ALL_COMPLETED, timeout=None
    ) if len(PLAN_MODELS) <= 3 else await asyncio.wait(
        plan_tasks, return_when=asyncio.FIRST_COMPLETED
    )

    # For >3 models: keep collecting until we have 3, then cancel the rest
    if len(PLAN_MODELS) > 3:
        completed: list = list(done)
        pending_set = set(pending)
        while len(completed) < 3 and pending_set:
            more_done, pending_set = await asyncio.wait(
                pending_set, return_when=asyncio.FIRST_COMPLETED
            )
            completed.extend(more_done)
        # Cancel any remaining (the slowest)
        for t in pending_set:
            t.cancel()
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass
        status(f"Layer 1: {len(completed)} finished first, cancelled {len(pending_set)} stragglers")
        done_tasks = completed
    else:
        done_tasks = list(done)

    # Collect results from winners
    plans = []
    for t in done_tasks:
        try:
            r = t.result()
            if isinstance(r, dict) and r.get("answer"):
                plans.append(r)
        except Exception:
            pass

    if not plans:
        raise RuntimeError("All models failed in planning")

    status(f"Layer 1: got {len(plans)} winning plans (research_cache: {len(research_cache)} entries)")

    all_plans_text = "\n\n".join(
        f"=== PLAN BY {p['model'].split('/')[-1].upper()} ===\n{p['answer']}"
        for p in plans
    )

    if extended:
        # == Layer 2: 4 Nemotron Super instances each read ALL plans, write their own improved plan ==
        step(f"Layer 2: 4 Nemotron Super instances improving {len(plans)} plans...")

        # Pre-load Layer 1 research so improvers don't re-search the same code
        preloaded_research = _format_research_cache(research_cache)

        improve_prompt = SYSTEM_KNOWLEDGE + f"""

You are a senior engineer. You have been given {len(plans)} independent plans
for a coding task. Your job:

1. Read ALL plans carefully
2. Find the FLAWS in each plan (logic errors, missing steps, wrong assumptions,
   broken edge cases, over-engineering, under-engineering)
3. Find the STRENGTHS in each plan (good ideas the others missed)
4. Write YOUR OWN IMPROVED PLAN that:
   - Takes the best approach from each plan
   - Fixes every flaw you found
   - Adds anything that ALL plans missed
   - Removes anything unnecessary

TASK: {task}

PROJECT:
{context[:15000]}

ALL PLANS FROM LAYER 1:
{all_plans_text}
{preloaded_research}
TOOLS (verify claims against real code):
  [REFS: name] / [LSP: name] / [DETAIL: feature] / [CODE: path]

FIRST write your analysis: what's good, what's bad, what's missing.
THEN write your improved plan.

COMPLETENESS CHECK: Re-read the TASK. Count every separate thing it asks for.
Your improved plan MUST address ALL of them.

RULES:
- NO CODE -- plain English only
- Your plan must be COMPLETE and SELF-CONTAINED
- Be detailed enough for a developer to implement without questions

YOUR IMPROVED PLAN:

## PLAN SUMMARY
## DETAILED STEPS
1. [File: path] [Section: name]
   Current behavior: ...
   New behavior: ...
   Logic: ...
   State/data: ...
   Connections: ...
## FILES TO MODIFY
## FILES TO CREATE (if any)
## EDGE CASES
## TEST CRITERIA
"""

        improved_results = list(await asyncio.gather(
            *[_call_with_tools("nvidia/nemotron-super", improve_prompt, project_root,
                               detailed_map=detailed_map, purpose_map=purpose_map,
                               research_cache=research_cache,
                               log_label="improving plan (Layer 2)")
              for _ in range(4)],
            return_exceptions=True,
        ))
        improved = [d for d in improved_results if isinstance(d, dict) and d.get("answer")]

        if not improved:
            warn("Layer 2: all failed, falling back to Layer 1 plans")
            improved = plans

        status(f"Layer 2: got {len(improved)} improved plans (cache: {len(research_cache)} entries)")


        all_improved_text = "\n\n".join(
            f"=== IMPROVED PLAN BY {d['model'].split('/')[-1].upper()} ===\n{d['answer']}"
            for d in improved
        )

        # == Layer 3: GLM-5 reads all improved plans, finds flaws/strengths, writes final ==
        step("Layer 3: GLM-5 writing final plan...")

        # Update pre-loaded research (now includes Layer 2's lookups too)
        preloaded_research = _format_research_cache(research_cache)

        verify_block = ""
        if not is_new_project:
            verify_block = (
                "Verify claims against real code:\n"
                "  [REFS: name] / [LSP: name] / [DETAIL: feature] / [CODE: path]\n"
                "  [REFS: name] — find all definitions, imports, usages\n"
            )

        merge_prompt = SYSTEM_KNOWLEDGE + f"""

YOUR ROLE: FINAL PLAN WRITER (Layer 3 of 3)

4 engineers each wrote a plan. Then 4 senior engineers each read ALL plans,
found flaws and strengths, and wrote their own improved version.
You now have {len(improved)} improved plans. Your job:

1. Read every improved plan
2. Find what each one gets RIGHT that others miss
3. Find FLAWS that still remain in any of them
4. Write THE FINAL plan: the best from all, with every flaw fixed

TASK: {task}

PROJECT:
{context[:10000]}

{verify_block}

ALL IMPROVED PLANS (Layer 2 output):
{all_improved_text[:30000]}
{preloaded_research}
RULES:
- NO CODE -- plain English only
- Be detailed enough for implementation without questions
- Stay focused on the TASK
- COMPLETENESS: Re-read the TASK. Your final plan MUST address every
  separate thing it asks for. If the task has 3 parts, you need all 3.
  Before finishing, verify each request from the TASK is covered.

Output the FINAL plan:

## PLAN SUMMARY
## DETAILED STEPS
1. [File: path] [Section: name]
   Current behavior: ...
   New behavior: ...
   Logic: ...
   State/data: ...
   Connections: ...
## FILES TO MODIFY
## FILES TO CREATE (if any)
## EDGE CASES
## TEST CRITERIA
"""

        merger_result = await _call_with_tools(
            "nvidia/glm-5", merge_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label="merging plans (final)")

    else:
        # == Standard: GLM-5 merges plans directly (no debate) ==
        step("Layer 2: GLM-5 merging plans...")

        # Pre-load Layer 1 research for the merger
        preloaded_research = _format_research_cache(research_cache)

        verify_block = ""
        if not is_new_project:
            verify_block = (
                "Verify claims against real code:\n"
                "  [REFS: name] / [LSP: name] / [DETAIL: feature] / [CODE: path]\n"
                "  [REFS: name] — find all definitions, imports, usages\n"
                "Write tags, wait, then proceed.\n"
            )

        merge_prompt = SYSTEM_KNOWLEDGE + f"""

YOUR ROLE: MERGER
Take the best from {len(plans)} independent plans. Combine, find flaws, improve.

TASK: {task}

PROJECT:
{context[:15000]}

{verify_block}

ALL PLANS:
{all_plans_text}
{preloaded_research}
INSTRUCTIONS:
1. Read every plan carefully
2. Identify good ideas in each
3. Find flaws: missing edge cases, logic errors, broken steps
4. Combine into one coherent plan
5. Fix everything you found
6. COMPLETENESS: Re-read the TASK. Does your merged plan address EVERY
   separate thing it asks for? If the task has multiple parts, cover all.

RULES:
- NO CODE -- plain English only
- Be detailed for implementation without questions
- Stay on task

Output the FINAL plan:

## PLAN SUMMARY
## DETAILED STEPS
1. [File: path] [Section: name]
   Current behavior: ...
   New behavior: ...
   Logic: ...
## FILES TO MODIFY
## FILES TO CREATE (if any)
## EDGE CASES
## TEST CRITERIA
"""

        merger_result = await _call_with_tools(
            "nvidia/glm-5", merge_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label="merging plans")

    if not merger_result.get("answer"):
        best = max(plans, key=lambda p: len(p["answer"]))
        return best["answer"], research_cache

    best_plan = merger_result["answer"]
    status(f"Phase 2: final plan = {len(best_plan)} chars")
    success(f"Phase 2 complete ({mode_label}, {len(research_cache)} cached lookups)")
    return best_plan, research_cache


# =====================================================================
#  PHASE 3 -- IMPLEMENT (parallel coders)
# =====================================================================

async def phase_implement(
    task: str, plan: str, context: str, sandbox: Sandbox,
    project_root: str, files_to_modify: list[str], detailed_map: str = "",
    purpose_map: str = "",
    research_cache: dict | None = None,
) -> tuple[str, Sandbox]:
    """
    Implement ALL files in parallel. Each coder gets the plan + its own file.
    Returns: (plan, sandbox_with_changes)
    """
    step("=== Phase 3: IMPLEMENT (parallel) ===")

    files_to_create = _extract_new_files_from_plan(plan)
    all_target_files = list(set(files_to_modify + files_to_create))
    if not all_target_files:
        all_target_files = files_to_modify if files_to_modify else ["main"]

    all_target_files = [
        os.path.relpath(f, project_root) if os.path.isabs(f) else f
        for f in all_target_files
    ]

    status(f"Target files ({len(all_target_files)}): {', '.join(all_target_files)}")

    async def _implement_one_file(target_file: str) -> tuple[str, str | None]:
        """Implement a single file. Returns (filepath, content) or (filepath, None).

        If the result has syntax errors, the broken code is sent back to the AI
        with the exact error so it can fix just the syntax — no rollback.
        """
        full_path = os.path.join(project_root, target_file)
        existing_content = sandbox.load_file(target_file) or read_file(full_path) or ""
        is_new = not existing_content

        MAX_RETRIES = 3

        for attempt in range(1, MAX_RETRIES + 1):
            step(f"Coder writing: {target_file} (attempt {attempt})")

            # Pre-load research cache so the coder already has planner lookups
            preloaded_research = _format_research_cache(research_cache)

            impl_prompt = IMPLEMENT_PROMPT.format(
                task=task,
                plan=plan[:12000],
                file_content=f"== {target_file} ==\n{existing_content}" if existing_content else "(new file)",
                context=context[:10000],
                target_file=target_file,
                cot_instructions=IMPL_COT_NEW if is_new else IMPL_COT_EXISTING,
            )

            # Inject pre-loaded research between the prompt and CoT instructions
            if preloaded_research:
                impl_prompt += preloaded_research

            impl_result = await _call_with_tools(
                IMPLEMENT_MODEL, impl_prompt, project_root,
                detailed_map=detailed_map, purpose_map=purpose_map,
                research_cache=research_cache,
                log_label=f"writing code: {target_file}",
            )

            extracted = _extract_code_blocks(impl_result["answer"])
            candidate_content = None

            if is_new:
                if extracted["new_files"]:
                    for fname, fcontent in extracted["new_files"].items():
                        candidate_content = fcontent
                        break
                if not candidate_content and extracted["edits"]:
                    replace_parts = []
                    for fp, pairs in extracted["edits"].items():
                        for find_text, replace_text in pairs:
                            if replace_text.strip():
                                replace_parts.append(replace_text.strip())
                    if replace_parts:
                        candidate_content = "\n\n".join(replace_parts)
                if not candidate_content:
                    raw_blocks = re.findall(r'```[^\n]*\n(.*?)```', impl_result["answer"], re.DOTALL)
                    if raw_blocks:
                        candidate_content = max(raw_blocks, key=len).strip()
            else:
                if extracted["edits"]:
                    edit_pairs = []
                    for fp, pairs in extracted["edits"].items():
                        edit_pairs.extend(pairs)
                    if edit_pairs:
                        candidate_content = _apply_edits(existing_content, edit_pairs)
                elif extracted["new_files"]:
                    for fname, fcontent in extracted["new_files"].items():
                        candidate_content = fcontent
                        break

            if not candidate_content:
                warn(f"  No {'content' if is_new else 'edits'} for {target_file} -- retrying")
                continue

            # ── Syntax check — fix in place on failure ─────────────────
            candidate_content = await _syntax_fix_loop(
                target_file, candidate_content, project_root,
                detailed_map=detailed_map, purpose_map=purpose_map,
                research_cache=research_cache,
            )

            if candidate_content is not None:
                label = "Created" if is_new else "Edited"
                status(f"  {label}: {target_file} ({len(candidate_content)} chars, syntax OK)")
                return target_file, candidate_content

            # _syntax_fix_loop returned None — unfixable, retry from scratch
            warn(f"  {target_file}: syntax unfixable, retrying full implementation")

        # All retries exhausted
        warn(f"  {target_file}: giving up after {MAX_RETRIES} attempts")
        return target_file, None

    # Run all file implementations in parallel
    results = await asyncio.gather(
        *[_implement_one_file(f) for f in all_target_files],
        return_exceptions=True,
    )

    final_files = {}
    for r in results:
        if isinstance(r, Exception):
            warn(f"File implementation failed: {r}")
            continue
        filepath, content = r
        if content:
            sandbox.write_file(filepath, content)
            final_files[filepath] = content

    success(f"Phase 3 complete -- {len(final_files)} files implemented (parallel)")
    return plan, sandbox


# =====================================================================
#  PHASE 3.5 -- REVIEW (GLM-5 checks code against plan, finds flaws)
# =====================================================================

async def phase_review(
    task: str, plan: str, sandbox: Sandbox,
    project_root: str, detailed_map: str = "",
    purpose_map: str = "",
    context: str = "",
    research_cache: dict | None = None,
) -> tuple[bool, Sandbox]:
    """
    GLM-5 reviews only the files that were ACTUALLY CHANGED in parallel.
    Finds flaws, missing pieces, improvements. Writes fixes.
    Returns: (had_fixes, sandbox)
    """
    step("=== Phase 3.5: CODE REVIEW (parallel per file) ===")

    # Only review files that were actually modified or created
    changed_files = {}
    for fp, content in sandbox.modified_files.items():
        changed_files[fp] = content
    for fp, content in sandbox.new_files.items():
        changed_files[fp] = content

    if not changed_files:
        status("No changed files to review")
        return False, sandbox

    status(f"Reviewing {len(changed_files)} changed file(s): {', '.join(changed_files.keys())}")

    async def _review_one_file(filepath: str, content: str) -> tuple[str, list]:
        """Review one file. Returns (filepath, list of edit pairs) or (filepath, [])."""
        step(f"Reviewing: {filepath}")

        # Show other changed files for context
        other_files_str = ""
        for fp, fc in changed_files.items():
            if fp != filepath:
                other_files_str += f"\n== {fp} (first 2000 chars) ==\n{fc[:2000]}\n"

        # Pre-load research cache so the reviewer already has planner+coder lookups
        preloaded_research = _format_research_cache(research_cache)

        review_prompt = SYSTEM_KNOWLEDGE + f"""

YOUR ROLE: CODE REVIEWER

Review this file that was just written by another AI.
Find flaws, missing pieces, and things that could better achieve the plan.

TASK: {task}

PLAN (what should have been implemented):
{plan[:8000]}

THIS FILE ({filepath}):
{content}

OTHER FILES CHANGED (for reference):
{other_files_str[:6000]}

PROJECT CONTEXT:
{context[:5000]}
{preloaded_research}
TOOLS — use these to check things before flagging issues:
  [REFS: name]          — find where a function is defined, imported, used
  [LSP: name]           — semantic deps, types
  [DETAIL: section]     — code map for a feature
  [CODE: path/file]     — read actual source code
  [SEARCH: pattern]     — ripgrep search
Write tags, then STOP.

SAFETY RULE:
Before suggesting ANY change that could affect code OUTSIDE this file
(changing a function signature, renaming something, changing a return value),
you MUST first use [LSP: name] to check dependencies. LSP shows semantic
connections — what depends on this, what types flow through it.
If you can't verify it's safe, do NOT suggest the change.

CHECK:
1. Does this file implement its part of the plan correctly?
2. Logic errors or bugs?
3. Missing error handling?
4. Edge cases not handled?
5. Missing imports?
6. If any function signature or name changed, use [LSP:] to check dependencies

OUTPUT:
If the file is good: write APPROVED

If there are issues, write EDIT blocks:
=== EDIT: {filepath} ===
[SEARCH]
code that needs fixing
[/SEARCH]
[REPLACE]
corrected code
[/REPLACE]

Only fix real issues. Do NOT refactor for style.
Do NOT change interfaces unless you've verified all callers with [REFS:].
"""

        result = await _call_with_tools(
            "nvidia/glm-5", review_prompt, project_root,
            detailed_map=detailed_map, purpose_map=purpose_map,
            research_cache=research_cache,
            log_label=f"reviewing: {filepath}",
        )
        answer = result.get("answer", "")

        if "APPROVED" in answer.upper() and "[SEARCH]" not in answer:
            status(f"  {filepath}: APPROVED")
            return filepath, []

        extracted = _extract_code_blocks(answer)
        edit_pairs = []
        if extracted["edits"]:
            for fp, pairs in extracted["edits"].items():
                edit_pairs.extend(pairs)

        status(f"  {filepath}: {len(edit_pairs)} fixes")
        return filepath, edit_pairs

    # Run all reviews in parallel
    results = await asyncio.gather(
        *[_review_one_file(fp, content) for fp, content in changed_files.items()],
        return_exceptions=True,
    )

    total_fixes = 0
    for r in results:
        if isinstance(r, Exception):
            warn(f"Review failed: {r}")
            continue
        filepath, edit_pairs = r
        if edit_pairs:
            existing = sandbox.load_file(filepath) or changed_files.get(filepath, "")
            if existing:
                modified = _apply_edits(existing, edit_pairs)
                # Syntax-check the reviewer's fixes — try to fix if broken
                fixed = await _syntax_fix_loop(
                    filepath, modified, project_root,
                    detailed_map=detailed_map, purpose_map=purpose_map,
                    research_cache=research_cache,
                )
                if fixed is not None:
                    sandbox.write_file(filepath, fixed)
                    total_fixes += len(edit_pairs)
                else:
                    warn(f"  Reviewer fix for {filepath} introduced unfixable syntax error")
                    status(f"  Keeping pre-review version of {filepath}")

    if total_fixes:
        success(f"Code review: applied {total_fixes} fixes across {len(changed_files)} files")
    else:
        success(f"Code review: all {len(changed_files)} files APPROVED")

    return total_fixes > 0, sandbox


# ═══════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — TEST (optional)
# ═══════════════════════════════════════════════════════════════════════════════

async def phase_test(test_command: str, project_root: str) -> dict:
    """Run tests. Only called if user asked for testing."""
    step("═══ Phase 4: TEST ═══")

    try:
        result = subprocess.run(
            test_command, shell=True, capture_output=True, text=True,
            timeout=120, cwd=project_root,
        )
        passed = result.returncode == 0
        output = result.stdout + "\n" + result.stderr

        if passed:
            success("Tests PASSED")
        else:
            warn(f"Tests FAILED (exit {result.returncode})")

        return {
            "passed": passed,
            "output": output[:10000],
            "exit_code": result.returncode,
        }
    except subprocess.TimeoutExpired:
        warn("Tests timed out (120s)")
        return {"passed": False, "output": "TIMEOUT after 120s", "exit_code": -1}
    except Exception as e:
        error(f"Test execution failed: {e}")
        return {"passed": False, "output": str(e), "exit_code": -1}


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

async def code_agent(state: AgentState) -> AgentState:
    """
    Full coding agent workflow.
    Expects state to have: raw_input, classification (with complexity, domain).
    Optionally: project_root in state or defaults to cwd.
    """
    step("═══ CODING AGENT ═══")

    task = state.get("processed_input", state["raw_input"])
    classification = state.get("classification", {})
    complexity = classification.get("complexity", 5)

    # Determine project root
    project_root = state.get("project_root", os.getcwd())
    status(f"Project: {project_root}")
    status(f"Complexity: {complexity}")

    # Check if user wants testing
    wants_test = any(kw in task.lower() for kw in ["test", "run test", "verify", "check if"])
    test_command = None

    # Extract test command if specified
    test_match = re.search(r'\[TEST:\s*(.+?)\]', task, re.IGNORECASE)
    if test_match:
        test_command = test_match.group(1)
        wants_test = True
        task = re.sub(r'\[TEST:\s*.+?\]', '', task).strip()

    sandbox = Sandbox(project_root)

    try:
        # ── Generate/load code maps ───────────────────────────────────
        from tools.code_index import generate_maps, list_sections, list_purposes
        maps = await generate_maps(project_root)
        general_map = maps["general"]
        detailed_map = maps["detailed"]
        purpose_map = maps.get("purpose", "")

        # Empty project: clear maps so AIs don't try to look up nothing
        is_new_project = (not detailed_map or detailed_map == "(empty project)")
        if is_new_project:
            detailed_map = ""
            general_map = ""
            purpose_map = ""

        files = []
        if detailed_map:
            sections = list_sections(detailed_map)
            status(f"Code map: {len(sections)} sections indexed")

        purposes = list_purposes(purpose_map) if purpose_map else []
        if purposes:
            status(f"Purpose map: {len(purposes)} categories indexed")

        # Build context from maps
        context_parts = []

        if general_map:
            context_parts.append(f"PROJECT OVERVIEW (general map):\n{general_map}")

            # List available sections
            if detailed_map and sections:
                section_list = "\n".join(f"  - {s}" for s in sections)
                context_parts.append(
                    f"AVAILABLE CODE SECTIONS (use [DETAIL: name] to expand):\n"
                    f"{section_list}"
                )

            # List available purpose categories
            if purposes:
                purpose_list = "\n".join(f"  - {p}" for p in purposes)
                context_parts.append(
                    f"AVAILABLE PURPOSE CATEGORIES (use [PURPOSE: name] to see all code for a purpose):\n"
                    f"{purpose_list}\n"
                    f"Each category returns ALL code snippets that serve that purpose,\n"
                    f"with 10 lines of context. Assume nothing else in the project serves\n"
                    f"that purpose beyond what's listed."
                )

            # Auto-inject relevant knowledge based on the task
            from knowledge import get_auto_inject, list_knowledge
            knowledge_text = get_auto_inject(task)
            if knowledge_text:
                context_parts.append(knowledge_text)

            # List available knowledge topics
            knowledge_topics = list_knowledge()
            if knowledge_topics:
                kl = ", ".join(knowledge_topics)
                context_parts.append(
                    f"AVAILABLE KNOWLEDGE (use [KNOWLEDGE: topic] to consult):\n"
                    f"  {kl}"
                )

            context_parts.append(
                "TOOLS — use in order, escalate only if you need more:\n"
                "  1. [REFS: name]          — definitions, imports, usages (fast)\n"
                "  2. [LSP: name]           — semantic deps, types (if REFS not enough)\n"
                "  3. [DETAIL: section]     — organized code map for a feature\n"
                "     [PURPOSE: category]   — all code for a purpose (e.g. 'UI colors')\n"
                "  4. [CODE: path/to/file]  — read actual source code (last resort)\n"
                "     [SEARCH: pattern]     — ripgrep search\n"
                "     [WEBSEARCH: query]    — web search for API docs\n"
                "  [KNOWLEDGE: topic]       — consult design/game/planning guidelines\n"
                "Write tags, then STOP on its own line."
            )
        else:
            context_parts.append(
                f"PROJECT: {project_root}\n"
                f"STATUS: New project — no existing code. Create all files from scratch.\n"
                f"Do NOT use [DETAIL:] or [CODE:] tags — there is nothing to look up.\n"
                f"Just write the complete implementation directly."
            )

        context = "\n\n".join(context_parts)

        # Create sandbox
        sandbox.setup()

        # ── Phase 2: PLAN ────────────────────────────────────────────────
        plan, research_cache = await phase_plan(
            task, context, complexity, project_root, "", detailed_map,
            purpose_map=purpose_map, is_new_project=is_new_project,
        )

        # Extract files to modify from plan
        files_to_modify = _extract_files_from_plan(plan, files)
        if not files_to_modify:
            files_to_modify = []
        status(f"Files to modify: {', '.join(files_to_modify) or '(new files)'}")
        status(f"Sharing {len(research_cache)} cached lookups with coders + reviewers")

        # ── Phase 3: IMPLEMENT (parallel coders, shared research) ────────
        final_plan, sandbox = await phase_implement(
            task, plan, context, sandbox, project_root, files_to_modify, detailed_map,
            purpose_map=purpose_map, research_cache=research_cache,
        )

        # ── Phase 3.5: CODE REVIEW (GLM-5 checks code against plan) ─────
        had_fixes, sandbox = await phase_review(
            task, plan, sandbox, project_root, detailed_map, purpose_map, context,
            research_cache=research_cache,
        )

        # ── Phase 4: TEST (optional) ─────────────────────────────────────
        if wants_test and test_command:
            test_result = await phase_test(test_command, project_root)
            if not test_result["passed"]:
                warn("Tests failed — including failure info in output")
                context += f"\n\nTEST FAILURE:\n{test_result['output']}"

        # ── Phase 5: DELIVER ─────────────────────────────────────────────
        step("═══ Phase 5: DELIVER ═══")

        diff = sandbox.get_all_diffs()
        file_summary = sandbox.summary()

        # Ask DeepSeek to summarize AND update maps in parallel
        step("DeepSeek summarizing + updating maps...")

        summary_task = _call(IMPLEMENT_MODEL, SUMMARY_PROMPT.format(
            task=task,
            files_changed=file_summary,
            diff=diff[:15000],
        ), max_tokens=2048, log_label="summarizing changes")

        map_update_task = _call(IMPLEMENT_MODEL, MAP_UPDATE_PROMPT.format(
            task=task,
            files_changed=file_summary,
            diff=diff[:15000],
            general_map=general_map[:8000],
            detailed_map=detailed_map[:30000],
        ), log_label="updating code maps")

        summary_result, map_result = await asyncio.gather(
            summary_task, map_update_task
        )

        ai_summary = summary_result["answer"] if summary_result.get("answer") else file_summary

        # Parse map edits from map_result and apply them
        updated_general = general_map  # fallback: keep current
        updated_detailed = detailed_map
        if map_result.get("answer"):
            raw = map_result["answer"]

            # Split response into general edits section and detailed edits section
            gen_match = re.search(
                r'===\s*GENERAL\s*MAP\s*EDITS\s*===(.*?)(?====\s*DETAILED\s*MAP\s*EDITS\s*===|$)',
                raw, re.DOTALL | re.IGNORECASE
            )
            det_match = re.search(
                r'===\s*DETAILED\s*MAP\s*EDITS\s*===(.*)',
                raw, re.DOTALL | re.IGNORECASE
            )

            if gen_match and "no changes" not in gen_match.group(1).lower()[:100]:
                updated_general = _apply_map_edits(general_map, gen_match.group(1))
            if det_match and "no changes" not in det_match.group(1).lower()[:100]:
                updated_detailed = _apply_map_edits(detailed_map, det_match.group(1))

        output = f"""## Changes Ready

{ai_summary}

Apply these changes to {project_root}? (y/n)"""

        state["final_answer"] = output
        state["pending_sandbox"] = sandbox
        state["updated_maps"] = {
            "general": updated_general,
            "detailed": updated_detailed,
        }
        success("Coding agent complete — waiting for user approval")

    except Exception as e:
        error(f"Coding agent failed: {e}")
        state["final_answer"] = f"Coding agent error: {e}\n\nPartial results may be in the sandbox."

    finally:
        # Don't cleanup sandbox — user might want to inspect
        pass

    return state


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_files_from_plan(plan: str, known_files: list[str]) -> list[str]:
    """Extract file paths from the plan text (files to MODIFY)."""
    files = set()

    for line in plan.split("\n"):
        matches = re.findall(r'[\w./\-]+\.(?:py|js|ts|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md)', line)
        files.update(matches)

    for f in known_files:
        if f in plan:
            files.add(f)

    return sorted(files)


def _extract_new_files_from_plan(plan: str) -> list[str]:
    """Extract files to CREATE from the plan (listed under FILES TO CREATE)."""
    files = []
    in_create_section = False

    for line in plan.split("\n"):
        line_stripped = line.strip()
        if re.match(r'#+\s*FILES?\s*TO\s*CREATE', line_stripped, re.IGNORECASE):
            in_create_section = True
            continue
        if in_create_section:
            if line_stripped.startswith("##") or (not line_stripped and files):
                break  # End of section
            matches = re.findall(r'[\w./\-]+\.(?:py|js|ts|html|css|json|lean|c|cpp|h|rs|java|go|rb|toml|yaml|yml|md)', line_stripped)
            files.extend(matches)

    return files


def _guess_filename(task: str, content: str) -> str:
    """Guess a filename from task description and file content."""
    task_lower = task.lower()
    content_start = content[:200].lower()

    # Detect from content
    if "<!doctype html" in content_start or "<html" in content_start:
        return "index.html"
    if content_start.strip().startswith("<!doctype"):
        return "index.html"
    if "import react" in content_start or "from react" in content_start:
        return "App.jsx"
    if "package main" in content_start:
        return "main.go"
    if "fn main" in content_start:
        return "main.rs"
    if "public class" in content_start or "public static void main" in content_start:
        return "Main.java"
    if "#include" in content_start:
        return "main.cpp" if "iostream" in content_start else "main.c"
    if "theorem " in content_start or "import Mathlib" in content_start:
        return "proof.lean"
    if content_start.strip().startswith("{"):
        return "data.json"

    # Detect from task
    if any(kw in task_lower for kw in ["html", "webpage", "website"]):
        return "index.html"
    if any(kw in task_lower for kw in [" css", "stylesheet"]):
        return "style.css"
    if any(kw in task_lower for kw in ["javascript", " js ", ".js"]):
        return "script.js"
    if any(kw in task_lower for kw in ["react", "component", "jsx"]):
        return "App.jsx"
    if any(kw in task_lower for kw in ["lean", "formal proof", "theorem prover"]):
        return "proof.lean"
    if any(kw in task_lower for kw in ["rust", "cargo"]):
        return "main.rs"
    if any(kw in task_lower for kw in ["web app", "web game", "for chrome", "browser game", "in browser"]):
        return "index.html"

    # Default to Python
    return "main.py"
