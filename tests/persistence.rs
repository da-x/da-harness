// Integration test exercising the persistence hooks (restore / save).
//
// Two tests:
// 1. `persistence_hooks_fire_correctly` — runs a fresh counter loop to target 3
//    and verifies that `restore` and `save` are called at the right times with
//    the right data.
// 2. `restore_from_middle_state` — pre-seeds the agent with synthetic history
//    so that `restore` returns mid-loop state (counter at 2), then verifies the
//    loop continues from that point and reaches target 3.
//
// The counter agent follows the same pattern as CountAgent in compaction.rs:
// the LLM increments a counter step-by-step then signals done when the target
// is reached. Few-shot examples make this reliable across models.
//
// Run with (env vars required, same as kv_agent):
//   DA_HARNESS_API_BASE=http://127.0.0.1:4242/v1 \
//   DA_HARNESS_MODEL_NAME=LocalModel \
//   DA_HARNESS_API_KEY="" \
//   cargo test --test persistence -- --nocapture

use std::collections::VecDeque;

use async_trait::async_trait;
use da_harness::{
    AgentLoop, LLMConfig, LoopConfigBuilder, LoopControl, OpenAIClient, RestoreState, SavePoint,
    run_loop,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::info;

// ─── Request / Response ────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, JsonSchema)]
struct CountRequest {
    /// The current counter value.
    current: u32,
    /// Target value to reach.
    target: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CountResponse {
    /// Ask the controller to increment the counter by one.
    increment: bool,
    /// Signal that the target has been reached and the loop should stop.
    done: bool,
}

// ─── Shared state to record hook invocations ──────────────────────────

/// Record of a single `save` call.
#[derive(Debug, Clone)]
struct SaveRecord {
    history_len: usize,
    has_summary: bool,
    current: u32,
    save_point: SavePoint,
}

/// Shared state between the agent and the test assertion.
struct PersistState {
    restore_called: bool,
    saves: VecDeque<SaveRecord>,
}

// ─── Agent ────────────────────────────────────────────────────────────

struct CountPersistAgent {
    state: PersistState,
    /// Optional pre-seeded restore payload.
    restore_payload: Option<RestoreState<CountRequest, CountResponse>>,
    target: u32,
}

#[async_trait]
impl AgentLoop for CountPersistAgent {
    type Request = CountRequest;
    type Response = CountResponse;
    type Output = u32;

    fn system_prompt(&self) -> &str {
        "You are a precise counter controller. Increment one step at a time, \
         then stop exactly when the target is reached."
    }

    fn user_prompt(&self) -> &str {
        "You control a counter. At each step you receive the current value and a target.\n\
         Output EXACTLY ONE raw JSON object conforming to the Response schema.\n\
         - Use increment=true to ask the controller to add 1.\n\
         - When current >= target, output done=true (and increment=false).\n\
         After a restore the prompt may contain a 'PREVIOUS SESSIONS SUMMARY' section; \
         use it to remember progress."
    }

    fn examples(&self) -> Vec<(Self::Request, Self::Response)> {
        vec![
            (
                CountRequest { current: 0, target: 2 },
                CountResponse {
                    increment: true,
                    done: false,
                },
            ),
            (
                CountRequest { current: 1, target: 2 },
                CountResponse {
                    increment: true,
                    done: false,
                },
            ),
            (
                CountRequest { current: 2, target: 2 },
                CountResponse {
                    increment: false,
                    done: true,
                },
            ),
        ]
    }

    fn initial_input(&self) -> Self::Request {
        CountRequest {
            current: 0,
            target: self.target,
        }
    }

    async fn iteration(
        &self,
        response: Self::Response,
        history: &[(Self::Request, Self::Response)],
    ) -> LoopControl<Self> {
        if response.done {
            let final_val = history.last().map(|(r, _)| r.current).unwrap_or(0);
            info!(final = final_val, "controller: done — stopping");
            return LoopControl::Stop(final_val);
        }

        if response.increment {
            let last = history.last().map(|(r, _)| r.current).unwrap_or(0);
            let next = last + 1;
            info!(from = last, to = next, "controller: increment");
            LoopControl::Continue(CountRequest {
                current: next,
                target: self.target,
            })
        } else {
            // LLM sent a no-op; feed the same current back.
            let last = history.last().map(|(r, _)| r.current).unwrap_or(0);
            LoopControl::Continue(CountRequest {
                current: last,
                target: self.target,
            })
        }
    }

    async fn restore(&mut self) -> Option<RestoreState<Self::Request, Self::Response>> {
        self.state.restore_called = true;
        self.restore_payload.take()
    }

    async fn save(
        &mut self,
        history: &[(Self::Request, Self::Response)],
        summary: Option<&str>,
        current_request: &Self::Request,
        save_point: SavePoint,
    ) {
        self.state.saves.push_back(SaveRecord {
            history_len: history.len(),
            has_summary: summary.is_some(),
            current: current_request.current,
            save_point,
        });
    }
}

// ─── Test helpers ─────────────────────────────────────────────────────

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

// ─── Test 1: fresh run — hooks fire at correct times ──────────────────

#[tokio::test]
async fn persistence_hooks_fire_correctly() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("da_harness::loop=info")
        .with_test_writer()
        .try_init();

    let client = client_from_env();
    let agent = CountPersistAgent {
        state: PersistState {
            restore_called: false,
            saves: VecDeque::new(),
        },
        restore_payload: None,
        target: 3,
    };

    let config = LoopConfigBuilder::default()
        .max_iterations(Some(10))
        .try_build()
        .unwrap();

    let (agent, output) = run_loop(client, agent, config).await.expect("agent loop failed");
    assert_eq!(output, 3, "expected counter to reach target 3");

    let final_state = &agent.state;

    assert!(final_state.restore_called, "restore() was not called");

    // Counter: current=0 (iter 0) → increment → current=1 (iter 1) → increment
    // → current=2 (iter 2) → increment → current=3 (iter 3) → done.
    // Save calls at iteration 1, 2, 3 (all Continue), then Final.
    let saves: Vec<_> = final_state.saves.iter().collect();

    assert_eq!(saves.len(), 4, "expected 4 save calls, got {}", saves.len());

    // Save #1 (iter 1): history has 1 entry, current=1.
    assert_eq!(saves[0].history_len, 1);
    assert!(!saves[0].has_summary);
    assert_eq!(saves[0].current, 1);
    assert_eq!(saves[0].save_point, SavePoint::Continue);

    // Save #2 (iter 2): history has 2 entries, current=2.
    assert_eq!(saves[1].history_len, 2);
    assert!(!saves[1].has_summary);
    assert_eq!(saves[1].current, 2);
    assert_eq!(saves[1].save_point, SavePoint::Continue);

    // Save #3 (iter 3): history has 3 entries, current=3.
    assert_eq!(saves[2].history_len, 3);
    assert!(!saves[2].has_summary);
    assert_eq!(saves[2].current, 3);
    assert_eq!(saves[2].save_point, SavePoint::Continue);

    // Save #4 (Final): history has 4 entries.
    assert_eq!(saves[3].history_len, 4);
    assert_eq!(saves[3].save_point, SavePoint::Final);
}

// ─── Test 2: restore from a middle checkpoint ─────────────────────────

#[tokio::test]
async fn restore_from_middle_state() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("da_harness::loop=info")
        .with_test_writer()
        .try_init();

    let client = client_from_env();

    // Pre-seeded checkpoint: counter at 2 (two increments already done).
    // History shows current=0→1 and current=1→2.
    let restore_payload = RestoreState {
        history: vec![
            (
                CountRequest { current: 0, target: 3 },
                CountResponse {
                    increment: true,
                    done: false,
                },
            ),
            (
                CountRequest { current: 1, target: 3 },
                CountResponse {
                    increment: true,
                    done: false,
                },
            ),
        ],
        current_request: CountRequest { current: 2, target: 3 },
        previous_summary: Some("Counter was incremented from 0 to 2. Target is 3.".to_string()),
    };

    let agent = CountPersistAgent {
        state: PersistState {
            restore_called: false,
            saves: VecDeque::new(),
        },
        restore_payload: Some(restore_payload),
        target: 3,
    };

    let config = LoopConfigBuilder::default()
        .max_iterations(Some(10))
        .try_build()
        .unwrap();

    let (agent, output) = run_loop(client, agent, config).await.expect("agent loop failed");
    assert_eq!(output, 3, "expected counter to reach target 3 from restored state");

    let final_state = &agent.state;

    assert!(final_state.restore_called, "restore() was not called");

    let saves: Vec<_> = final_state.saves.iter().collect();

    // Restored with 2 history entries, current=2. The loop sends current=2 to the
    // LLM (iter 0). The LLM increments (2 < 3), so current becomes 3. Iteration 1
    // fires save with history_len=3 (2 restored + 1 new), current=3. Then iter 1
    // sends current=3 to the LLM which responds done=true. Final save fires.
    // Total: 2 saves (Continue at iter 1, then Final).
    assert_eq!(saves.len(), 2, "expected 2 save calls, got {}", saves.len());

    // Save #1 (iter 1, Continue): history has 3 entries, summary present.
    assert_eq!(saves[0].history_len, 3);
    assert!(saves[0].has_summary, "expected the restored summary to be present");
    assert_eq!(saves[0].current, 3);
    assert_eq!(saves[0].save_point, SavePoint::Continue);

    // Save #2 (Final): history has 4 entries.
    assert_eq!(saves[1].history_len, 4);
    assert!(saves[1].has_summary, "expected the restored summary to still be present");
    assert_eq!(saves[1].save_point, SavePoint::Final);
}
