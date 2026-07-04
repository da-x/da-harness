# da-harness

A framework for running LLM-driven agent loops with tool calling and structured JSON request/response schemas.

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

## Overview

`da-harness` provides abstractions over OpenAI-compatible chat APIs to build agents that operate in iterative loops. It offers two patterns:

- **Multi-tool agents** (`multi_tool`) — The LLM drives itself by calling named tools you define. User messages are injected via an async channel, and the agent decides when to call tools or respond with text.
- **Typed request/response loops** (`single_tool`) — You define `Request`, `Response`, and `Output` types. Each iteration the agent returns structured JSON, your controller executes it, and feeds the result back as the next request.

Key features:

- **Tool calling** — Define tools with typed argument structs; schemas are auto-generated via `schemars`.
- **Parallel or serial execution** — Multiple tool calls from a single LLM response run concurrently or in order.
- **Typed agent contracts** — For `single_tool`, define `Request`, `Response`, and `Output` types automatically serialized to/from JSON.
- **Schema-aware prompts** — JSON schemas are embedded in prompts with shared type definitions deduplicated.
- **Few-shot examples** — Provide example request/response pairs formatted into the prompt automatically (`single_tool`).
- **Conversation history** — Full history included each iteration so the LLM has context.
- **Automatic context compaction** — When prompt usage approaches the model's context window, the loop automatically summarizes earlier turns and truncates history (`single_tool`).
- **Health checking** — Built-in endpoint health checks and readiness polling for `OpenAIClient`.

## Quick Start

Add `da-harness` to your `Cargo.toml`:

```toml
[dependencies]
da-harness = { git = "https://github.com/da-x/da-harness", branch = "r/0.3" }
```

### Multi-Tool Agent

Define tools with typed argument structs and an async handler, then build the agent with `AgentInvocationArgs`:

```rust
use std::sync::Arc;
use da_harness::multi_tool::{Tool, AgentInvocationArgs};
use da_harness::OpenAIClient;
use serde::Deserialize;
use schemars::JsonSchema;
use futures::FutureExt;

/// Doc about the tool
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct CalcArgs {
    /// Doc the parameter
    expression: String,
}

let calc_tool = Tool::new::<CalcArgs>(Arc::new(|args| {
    let expr = args.expression;
    async move {
        let result = eval(&expr)?;
        Ok(format!("Result: {}", result))
    }.boxed()
}))?;

let (tx, rx) = tokio::sync::mpsc::channel(32);
let invocation = AgentInvocationArgs::default()
    .system_prompt("You are a helpful assistant with access to tools.")
    .tools(vec![calc_tool])
    .parallel_tools(true)
    .incoming(rx)
    .agent_message_callback(|msg| Box::pin(async move {
        println!("Agent says: {}", msg);
        Ok(())
    }))
    .agent_idle_callback(|| Box::pin(async move {
        println!("Agent is idle awaiting user input");
        Ok(())
    }))
    .build()
    .unwrap();

// tx close ends the session
invocation.run(client).await?;
```

The loop runs until the incoming channel is closed. The LLM decides whether to call tools or produce a text response. Tool calls are resolved by your handlers, and results are fed back to the LLM.


### Typed Request/Response Agent

For agents with a structured request/response cycle, implement the `AgentLoop` trait:

```rust
use da_harness::single_tool::{AgentLoop, LoopConfigBuilder, LoopControl, run_loop};
use da_harness::OpenAIClient;
use async_trait::async_trait;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, JsonSchema)]
struct MyRequest {
    step: u32,
    question: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct MyResponse {
    answer: String,
    done: bool,
}

struct MyAgent;

#[async_trait]
impl AgentLoop for MyAgent {
    type Request = MyRequest;
    type Response = MyResponse;
    type Output = String;

    fn system_prompt(&self) -> &str {
        "You are a helpful assistant."
    }

    fn user_prompt(&self) -> &str {
        "Answer the following question. Set done to true when finished."
    }

    fn examples(&self) -> Vec<(Self::Request, Self::Response)> {
        vec![]
    }

    fn initial_input(&self) -> Self::Request {
        MyRequest { step: 1, question: "What is 6x7?".into() }
    }

    async fn iteration(
        &self,
        response: Self::Response,
        _history: &[(Self::Request, Self::Response)],
    ) -> LoopControl<Self> {
        if response.done {
            LoopControl::Stop(response.answer)
        } else {
            LoopControl::Continue(MyRequest {
                step: 2,
                question: "Now square that result.".into(),
            })
        }
    }
}

#[tokio::main]
async fn main() {
    let config = LoopConfigBuilder::default().try_build().unwrap();
    let client = OpenAIClient::new();
    let output = run_loop(client, MyAgent, config).await.unwrap();
    println!("Result: {}", output);
}
```

## Architecture

### Multi-Tool Agents

```
┌─────────────┐         ┌──────────────┐         ┌──────────────┐
│  Your Code  │ send    │   Agent      │ tool    │ LLM Endpoint │
│ (User Input)│────────►│  Loop        │◄───────►│ (OpenAI API) │
│             │<════════│  (multi_tool)│ text    │              │
└─────────────┘  result └──────────────┘ response└──────────────┘
```

1. Define tools using `Tool::new()`, each with a typed argument struct and an async handler.
2. Build an `AgentInvocation` via `AgentInvocationArgs` with your system prompt, tool list, and an incoming message channel.
3. Call `run()` — the loop runs until the incoming channel is closed.
4. The LLM decides whether to call tools or produce a text response. Tool calls are resolved by your handlers, and results are fed back to the LLM.

When the LLM requests multiple tool calls, they can be executed in two modes:

- **Parallel** (`parallel_tools(true)`) — All tool calls run concurrently via `futures::join_all`. Results are collected and returned to the LLM together.
- **Serial** (`parallel_tools(false)`) — Tool calls execute one at a time, in order. Each result is appended before the next call runs.

If a tool handler returns an error, the error message is still relayed to the LLM so it can recover.

### Typed Request/Response Agents

```
┌─────────────┐          ┌──────────┐         ┌──────────────┐
│  Your Code  │◄─────────│  Agent   │         │ LLM Endpoint │
│ (Controller)│ continue │  Loop    │◄────────│ (OpenAI API) │
│             │─────────►│  Trait   │────────►│              │
│  Executes   │  output  │          │ prompt  │              │
│  actions    │          │          │         │              │
└─────────────┘          └──────────┘         └──────────────┘
```

1. **`run_loop`** builds a prompt from the agent's system prompt, user prompt, JSON schemas, examples, conversation history, and current request.
2. The prompt is sent to the LLM via `OpenAIClient::chat`.
3. The LLM response is deserialized into the agent's `Response` type.
4. **`AgentLoop::iteration`** processes the response and returns either:
   - `LoopControl::Continue(next_request)` — executes your controller logic and feeds the result back as the next iteration's input.
   - `LoopControl::Stop(output)` — ends the loop and returns the final output value.

By default there is no iteration cap (the loop runs until the agent returns `Stop`). Use [`LoopConfigBuilder`] to set an optional `max_iterations` limit and configure compaction behavior.

## Context Compaction

Long-running agents can exceed the model's context window as conversation history grows. `da-harness` handles this automatically for typed request/response loops:

1. **Discovery** — On loop start, the framework queries the server's `/models` endpoint (or reads `LLMConfig::max_context_tokens`) to learn the model's maximum context window in tokens.
2. **Monitoring** — After each LLM response, the framework checks the `prompt_tokens` count reported by the server.
3. **Trigger** — When prompt usage reaches **75%** of the known window, a separate summarization call is issued. The LLM produces a compact summary targeting ~25% of the window.
4. **Rewrite** — The summary is injected into subsequent prompts under the `PREVIOUS SESSIONS SUMMARY` heading. Old typed history entries are dropped (the most recent few turns are kept verbatim).

Compaction is best-effort: if the summarization call fails, a warning is logged and the loop continues with the current history. The compaction call itself does not count against the iteration limit.

### Configuration

Control compaction via [`CompactPolicy`] on your [`LoopConfigBuilder`]:

```rust
// Default: auto-discover context window from the server.
let config = LoopConfigBuilder::default().try_build().unwrap();

// Disable compaction entirely.
let config = LoopConfigBuilder::default()
    .compact_policy(CompactPolicy::NoCompact)
    .try_build()
    .unwrap();

// Force a specific window size (useful for testing).
let config = LoopConfigBuilder::default()
    .compact_policy(CompactPolicy::Override(700))
    .try_build()
    .unwrap();
```

## Callbacks (Multi-Tool)

| Callback | When Called |
|---|---|
| `agent_message_callback` | The LLM produces a text response (no tool calls) |
| `agent_idle_callback` | No pending user messages and no prior tool call — the agent is waiting for input |

Both callbacks are optional. If not set, they default to no-ops.

## Prompt Structure

Each iteration sends a prompt with these sections:

| Section | Description |
|---|---|
| User Prompt | Your agent's instructions (from `user_prompt()`) |
| JSON Schema | Combined `Request` and `Response` schemas with shared definitions |
| Examples | Few-shot example pairs (from `examples()`) |
| Previous Sessions Summary | Optional compact summary of earlier turns (produced automatically by context compaction when the window limit is known) |
| Conversation History | Prior request/response pairs from the loop (may be truncated after compaction) |
| Additional Context | Dynamic content from `extend_prompt()` (optional) |
| Current Input | The serialized current request as JSON |

## API Reference

 <!-- See the [Rust documentation](https://docs.rs/da-harness) for the full API reference.  -->

Local docs can be generated with:

```bash
cargo doc --open
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
