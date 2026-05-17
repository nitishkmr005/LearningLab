# 07 — Agents

Exhaustive learning path for LLM-powered agents: reasoning, tool use, memory, planning, and multi-agent systems.

---

## 01 — ReAct: Reasoning + Acting
Thought → Action → Observation loop; implement from scratch with any LLM; trace and debug runs.
- https://arxiv.org/abs/2210.03629
- https://react-lm.github.io/

## 02 — Tool Use via Function Calling
JSON schema tool definitions; LLM selects tool + args; execute + inject observation; multi-tool turns.
- https://docs.anthropic.com/en/docs/tool-use
- https://platform.openai.com/docs/guides/function-calling

## 03 — Minimal Agent Loop (from scratch)
System prompt design; tool dispatch in Python; stopping condition; no framework, pure loop.
- https://www.anthropic.com/research/building-effective-agents

## 04 — Structured Outputs for Agents
Force schema-valid JSON with Instructor / Outlines; Pydantic model extraction; avoid regex parsing.
- https://python.useinstructor.com/
- https://github.com/outlines-dev/outlines

## 05 — Short-Term Memory: Conversation Buffer
In-context history; sliding window trimming; token budget management; summarization buffer.
- https://python.langchain.com/docs/how_to/chatbots_memory/

## 06 — Long-Term Memory: Vector Store
Persist episodic memories as embeddings; semantic search on retrieval; write/read memory hooks.
- https://python.langchain.com/docs/how_to/vectorstore_retriever/

## 07 — Structured Long-Term Memory: MemGPT / Letta
Main context + external memory pages; agent manages its own memory; entity tracking.
- https://arxiv.org/abs/2310.08560

## 08 — Chain-of-Thought Prompting
Zero-shot CoT ("think step by step"); few-shot CoT with examples; self-consistency (majority vote).
- https://arxiv.org/abs/2201.11903
- https://arxiv.org/abs/2203.11171

## 09 — Tree of Thoughts (ToT)
Branch reasoning tree; BFS/DFS search; score nodes with LLM evaluator; backtracking.
- https://arxiv.org/abs/2305.10601

## 10 — Reflection & Self-Critique (Reflexion)
Agent critiques own output; verbal reinforcement; iterative refinement; memory of past failures.
- https://arxiv.org/abs/2303.11366

## 11 — Plan-and-Execute Agents
Generate step-by-step plan upfront; execute each step; replan on failure; LangChain PlanAndExecute.
- https://arxiv.org/abs/2305.04091

## 12 — Code Execution Agent
LLM writes Python → sandbox execution (E2B / subprocess) → observe output/errors → fix iteratively.
- https://e2b.dev/docs
- https://jupyter-client.readthedocs.io/en/stable/

## 13 — Web Search Agent
Tavily / Brave / SerpAPI tools; multi-hop web research; cite sources; handle search failures.
- https://tavily.com/
- https://python.langchain.com/docs/integrations/tools/tavily_search/

## 14 — Multi-Agent: Orchestrator + Specialists
Orchestrator routes subtasks; specialist agents run in parallel or sequence; result aggregation.
- https://www.anthropic.com/research/building-effective-agents

## 15 — AutoGen Multi-Agent Framework
Conversational agent groups; human proxy; code execution; teachable agents; group chat.
- https://arxiv.org/abs/2308.08155
- https://microsoft.github.io/autogen/

## 16 — CrewAI: Role-Based Agents
Agent roles + goals; sequential/hierarchical process; task delegation; per-agent tools.
- https://docs.crewai.com/introduction

## 17 — LangGraph: Stateful Agent Graphs
Nodes + edges + typed state; cycles for iterative refinement; conditional routing; subgraphs.
- https://langchain-ai.github.io/langgraph/tutorials/introduction/

## 18 — Parallelization: Fan-Out & Map-Reduce
Parallel tool calls; map over document chunks; aggregate results; async concurrency with asyncio.
- https://www.anthropic.com/research/building-effective-agents

## 19 — Human-in-the-Loop
Interrupt before risky tool calls; approval gates; correction injection; LangGraph interrupt nodes.
- https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/

## 20 — Observability: LangSmith / Langfuse
Trace agent runs end-to-end; token + cost tracking; feedback labels; debugging tool failures.
- https://docs.smith.langchain.com/
- https://langfuse.com/docs

## 21 — Agent Evaluation
Task completion rate; trajectory correctness; tool selection accuracy; LLM-as-judge; AgentBench.
- https://arxiv.org/abs/2308.03688
- https://arxiv.org/abs/2310.03744

## 22 — Prompt Injection & Agent Safety
Direct and indirect injection; tool abuse; sandboxed execution; output filtering; guardrails.
- https://arxiv.org/abs/2302.12173
- https://github.com/guardrails-ai/guardrails

## 23 — Cost & Latency Optimization
Minimize tool calls; cache repeated lookups; choose smaller models for simple steps; batching.
- https://arxiv.org/abs/2305.08130

## 24 — Model Context Protocol (MCP)
Anthropic's open standard for exposing tools/resources to LLMs; MCP servers (local + remote); MCP clients; resources, tools, and prompts primitives; replacing bespoke integrations; growing ecosystem.
- https://modelcontextprotocol.io/introduction
- https://docs.anthropic.com/en/docs/build-with-claude/mcp

## 25 — Computer Use & GUI Agents
Anthropic computer use API; screenshot + action (click/type/scroll) loop; Playwright-based browser agents; SWE-bench for code agent evaluation; real-world automation use cases; safety constraints.
- https://docs.anthropic.com/en/docs/build-with-claude/computer-use
- https://arxiv.org/abs/2310.06770
