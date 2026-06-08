// Integration test: exercises the full AgentLoop + run_loop machinery against a live LLM.
//
// Scenario: The LLM drives reads/writes on an in-memory HashMap<String, String>.
// It must read a value, compute twice that number, store it under a new key, then finish.
//
// Run with (env vars required):
//   DA_HARNESS_API_BASE=http://127.0.0.1:4242/v1 \
//   DA_HARNESS_MODEL_NAME=LocalModel \
//   DA_HARNESS_API_KEY="" \
//   cargo test --test kv_agent -- --nocapture

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use da_harness::{AgentLoop, LLMConfig, LoopConfigBuilder, LoopControl, OpenAIClient, run_loop};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::info;

// ─── Request (controller → LLM) ──────────────────────────────────────

#[derive(Debug, Clone, Serialize, JsonSchema)]
#[serde(tag = "type")]
enum KvRequest {
    Start { instruction: String },
    GetResult { key: String, value: Option<String> },
    SetAck { key: String, ok: bool },
}

// ─── Response (LLM → controller) ─────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
enum KvResponse {
    Get { key: String },
    Set { key: String, value: String },
    Finish { note: String },
}

// ─── Agent implementation ────────────────────────────────────────────

struct KvAgent {
    store: Arc<Mutex<HashMap<String, String>>>,
    instruction: String,
}

#[async_trait]
impl AgentLoop for KvAgent {
    type Request = KvRequest;
    type Response = KvResponse;
    type Output = String;

    fn system_prompt(&self) -> &str {
        "You are a helpful assistant that controls a key-value store."
    }

    fn user_prompt(&self) -> &str {
        r#"You control a key-value store. You can read and write string values by key.

At each step, output exactly one JSON action conforming to the Response schema.
The controller will execute your action and respond with a Request containing the result.

When you have completed all required work, emit a "Finish" action with a brief note.
"#
    }

    fn examples(&self) -> Vec<(Self::Request, Self::Response)> {
        vec![
            (
                KvRequest::Start {
                    instruction: "What is stored under key 'x'?".into(),
                },
                KvResponse::Get { key: "x".into() },
            ),
            (
                KvRequest::GetResult {
                    key: "x".into(),
                    value: Some("hello".into()),
                },
                KvResponse::Set {
                    key: "y".into(),
                    value: "world".into(),
                },
            ),
        ]
    }

    fn initial_input(&self) -> Self::Request {
        KvRequest::Start {
            instruction: self.instruction.clone(),
        }
    }

    async fn iteration(
        &self,
        response: Self::Response,
        _history: &[(Self::Request, Self::Response)],
    ) -> LoopControl<Self> {
        match response {
            KvResponse::Get { key } => {
                let value = self.store.lock().unwrap().get(&key).cloned();
                info!(%key, found = %value.is_some(), "controller: Get");
                LoopControl::Continue(KvRequest::GetResult { key, value })
            }
            KvResponse::Set { key, value } => {
                self.store.lock().unwrap().insert(key.clone(), value);
                info!(%key, "controller: Set");
                LoopControl::Continue(KvRequest::SetAck { key, ok: true })
            }
            KvResponse::Finish { note } => {
                info!(%note, "controller: Finish — stopping loop");
                LoopControl::Stop(note)
            }
        }
    }
}

// ─── Test helpers ────────────────────────────────────────────────────

fn client_from_env() -> OpenAIClient {
    let api_base = std::env::var("DA_HARNESS_API_BASE").expect(
        "DA_HARNESS_API_BASE must be set to run integration tests \
         (e.g. http://127.0.0.1:4242/v1)",
    );
    let model_name =
        std::env::var("DA_HARNESS_MODEL_NAME").expect("DA_HARNESS_MODEL_NAME must be set");
    let api_key = std::env::var("DA_HARNESS_API_KEY").unwrap_or_default();

    OpenAIClient::with_config(LLMConfig {
        api_base,
        model_name,
        api_key,
        max_context_tokens: None,
    })
}

// ─── Integration test ────────────────────────────────────────────────

#[tokio::test]
async fn kv_read_modify_finish() {
    // Enable tracing so raw LLM I/O is visible with --nocapture.
    let _ = tracing_subscriber::fmt()
        .with_env_filter("da_harness::loop=info")
        .with_test_writer()
        .try_init();

    let client = client_from_env();

    // Seed the store. The LLM should read "21" from "target", compute 42, write it to "doubled".
    let store: Arc<Mutex<HashMap<String, String>>> =
        Arc::new(Mutex::new(HashMap::from([("target".into(), "21".into())])));

    let agent = KvAgent {
        store: store.clone(),
        instruction: "Read the value under key 'target'. Compute twice that number as a string and store it under 'doubled'. Then Finish.".into(),
    };

    let config = LoopConfigBuilder::default()
        .max_iterations(Some(10))
        .try_build()
        .unwrap();
    let (_agent, _output) = run_loop(client, agent, config).await.expect("agent loop failed");

    // Verify the LLM performed the correct operations.
    let final_map = store.lock().unwrap();
    assert_eq!(
        final_map.get("doubled"),
        Some(&"42".to_string()),
        "Expected key 'doubled' to be '42', but got: {:?}",
        final_map.get("doubled")
    );
}
