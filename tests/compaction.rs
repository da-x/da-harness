// Integration test exercising automatic context compaction.
//
// The test forces a very small context window (via the explicit parameter to
// run_loop_with_max_and_context) so that the 75% threshold is crossed after only
// a few iterations. The agent performs a multi-step counting task that would
// normally overflow a tiny window; compaction must produce a usable "previous
// sessions summary" and truncate history while still allowing the agent to
// reach the correct final result.
//
// Run with (env vars required, same as kv_agent):
//   DA_HARNESS_API_BASE=http://127.0.0.1:4242/v1 \
//   DA_HARNESS_MODEL_NAME=LocalModel \
//   DA_HARNESS_API_KEY="" \
//   cargo test --test compaction -- --nocapture

use async_trait::async_trait;
use da_harness::{
    AgentLoop, CompactPolicy, LLMConfig, LoopConfigBuilder, LoopControl, OpenAIClient, run_loop,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tracing::info;

// ─── Request / Response for the counting agent ────────────────────────

#[derive(Debug, Clone, Serialize, JsonSchema)]
struct CountRequest {
    /// The current counter value known to the controller before this turn.
    current: u32,
    /// Hint / instruction visible to the LLM (static or updated by controller).
    hint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
struct CountResponse {
    /// Ask the controller to increment the counter by one.
    increment: bool,
    /// Signal that the target has been reached and the loop should stop.
    done: bool,
}

// ─── Agent ────────────────────────────────────────────────────────────

struct CountAgent {
    target: u32,
}

#[async_trait]
impl AgentLoop for CountAgent {
    type Request = CountRequest;
    type Response = CountResponse;
    type Output = u32;

    fn system_prompt(&self) -> &str {
        "You are a precise counter controller. You must reach a numeric target by \
         repeatedly asking for single increments, then stopping exactly when the \
         target is reached or exceeded. Never skip values."
    }

    fn user_prompt(&self) -> &str {
        "You control a counter. At each step you receive the current value and a hint.\n\
         Output EXACTLY ONE raw JSON object conforming to the Response schema. \
         - Use increment=true to ask the controller to add 1 to the current value.\n\
         - When current >= target, output done=true (and increment=false).\n\
         The controller will feed the updated value back as the next request.\n\
         After compaction the prompt may contain a 'PREVIOUS SESSIONS SUMMARY' section; \
         use it (plus the recent CONVERSATION HISTORY) to remember progress."
    }

    fn examples(&self) -> Vec<(Self::Request, Self::Response)> {
        vec![
            (
                CountRequest {
                    current: 0,
                    hint: "Reach at least 2".into(),
                },
                CountResponse {
                    increment: true,
                    done: false,
                },
            ),
            (
                CountRequest {
                    current: 2,
                    hint: "Reach at least 2".into(),
                },
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
            hint: format!(
                "Increment one step at a time until current reaches at least {}. \
                 Only set done=true when current >= target.",
                self.target
            ),
        }
    }

    async fn iteration(
        &self,
        response: Self::Response,
        history: &[(Self::Request, Self::Response)],
    ) -> LoopControl<Self> {
        if response.done {
            let final_val = history.last().map(|(r, _)| r.current).unwrap_or(0);
            info!(final = final_val, "controller: done received — stopping");
            return LoopControl::Stop(final_val);
        }

        if response.increment {
            let last = history.last().map(|(r, _)| r.current).unwrap_or(0);
            let next = last + 1;
            info!(from = last, to = next, "controller: increment");
            LoopControl::Continue(CountRequest {
                current: next,
                hint: "continue or done if target reached".into(),
            })
        } else {
            // LLM sent a no-op; feed the same current back so it can decide again.
            let last = history.last().map(|(r, _)| r.current).unwrap_or(0);
            LoopControl::Continue(CountRequest {
                current: last,
                hint: "please choose increment or done".into(),
            })
        }
    }
}

// ─── Test helpers (same pattern as kv_agent) ──────────────────────────

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
        max_context_tokens: None, // we force via the run_loop param below
    })
}

// ─── Integration test ────────────────────────────────────────────────

#[tokio::test]
async fn compaction_forces_summary_and_truncation() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("da_harness::loop=info")
        .with_test_writer()
        .try_init();

    let client = client_from_env();

    // A deliberately tiny context window. The first turn alone used ~554 tokens
    // in the observed run. 75% of 700 is 525, so compaction will trigger after
    // the first response (history will be rewritten with a "PREVIOUS SESSIONS SUMMARY"
    // and truncated). This exercises the full compaction machinery.
    let config = LoopConfigBuilder::default()
        .max_iterations(Some(30))
        .compact_policy(CompactPolicy::Override(700))
        .try_build()
        .unwrap();

    // Target high enough to require multiple steps even after early compaction.
    let agent = CountAgent { target: 10 };

    let (_agent, final_count) = run_loop(client, agent, config)
        .await
        .expect("agent loop with forced compaction should succeed");

    assert_eq!(
        final_count, 10,
        "expected to reach target 10 via increments (possibly after compaction), got {}",
        final_count
    );
}
