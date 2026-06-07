# da-harness

A framework for running LLM-driven agent loops with structured JSON request/response schemas.

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)

## Overview

`da-harness` provides a simple abstraction over OpenAI-compatible chat APIs to build agents that operate in iterative loops. Each iteration the agent sends a structured JSON response, your controller executes it, and feeds the result back as the next request. The loop continues until the agent signals completion.

Key features:

- **Typed agent contracts** — Define `Request`, `Response`, and `Output` types that are automatically serialized to/from JSON.
- **Schema-aware prompts** — JSON schemas are generated from your types via `schemars` and embedded in prompts, with shared type definitions deduplicated.
- **Few-shot examples** — Provide example request/response pairs that are formatted into the prompt automatically.
- **Conversation history** — Full request/response history is included in each iteration so the LLM has context.
- **Extensible prompts** — Use `extend_prompt` to inject dynamic context based on conversation state.
- **Health checking** — Built-in endpoint health checks and readiness polling for `OpenAIClient`.

## Quick Start

Add `da-harness` to your `Cargo.toml`:

```toml
[dependencies]
da-harness = { git = "https://github.com/da-x/da-harness", branch = "r/0.1" }
```

Define your agent by implementing the `AgentLoop` trait:

```rust
use da_harness::{AgentLoop, LoopControl, OpenAIClient, run_loop};
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
    let client = OpenAIClient::new();
    let output = run_loop(client, MyAgent).await.unwrap();
    println!("Result: {}", output);
}
```

## Architecture

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

The default maximum iteration count is 10. Use `run_loop_with_max` (or `run_loop_with_max_and_context`) to customize the iteration limit. When a maximum context window size is known (via `LLMConfig`, an explicit parameter, or by querying the server's `/models` endpoint), the loop will automatically trigger history compaction once ~75% of the window has been used, asking the LLM to produce a compact "previous sessions summary" targeting ~25% of the window and truncating the retained typed history.

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
