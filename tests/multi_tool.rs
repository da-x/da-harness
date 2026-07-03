use std::sync::{Arc, Mutex};

use da_harness::multi_tool::{AgentInvocation, TaskFuture, Tool};
use da_harness::{LLMConfig, OpenAIClient};
use schemars::JsonSchema;
use serde::Deserialize;
use tracing::info;

/// Adds the given value to the running counter.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct Add {
    /// The amount to add to the counter.
    value: u32,
}

/// Stops the agent with the final counter result. Only call once counting is complete.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
struct Stop {
    /// The expected final counter value.
    result: u32,
}

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

#[tokio::test]
async fn multi_tool_count_to_target() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("da_harness=debug,multi_tool=info")
        .with_test_writer()
        .try_init();

    let client = client_from_env();
    let counter = Arc::new(Mutex::new(0u32));

    let counter_add = counter.clone();
    let add_tool = Tool::new(Arc::new(move |args: Add| -> TaskFuture {
        let c = counter_add.clone();
        Box::pin(async move {
            let mut val = c.lock().unwrap();
            *val += args.value;
            Ok(())
        })
    }))
    .unwrap();

    // When Stop is called, signal the stop_rx channel so the test can drop tx.
    let (stop_tx, mut stop_rx) = tokio::sync::mpsc::channel::<()>(1);

    let counter_stop = counter.clone();
    let stop_tx_inner = stop_tx.clone();
    let stop_tool = Tool::new(Arc::new(move |args: Stop| -> TaskFuture {
        let c = counter_stop.clone();
        let stx = stop_tx_inner.clone();
        Box::pin(async move {
            let val = *c.lock().unwrap();
            tracing::info!(counter = val, requested = args.result, "Stop tool called");
            let _ = stx.send(()).await;
            Ok(())
        })
    }))
    .unwrap();

    let (tx, rx) = tokio::sync::mpsc::channel(32);

    let invocation = AgentInvocation {
        system_prompt: "You are a counter agent. You have two tools:\n\
                       - Add(value): adds a value to the counter.\n\
                       - Stop(result): stops with the final result.\n\
                       Count from 0 to 5 by calling Add(1) five times, then call Stop(result=5).\n\
                       You MUST call Stop after exactly 5 Add calls. Do not keep calling Add indefinitely."
            .to_string(),
        tools: vec![add_tool, stop_tool],
        parallel_tools: false,
        incoming: rx,
        agent_message_callback: Arc::new(|msg: String| {
            Box::pin(async move {
                tracing::info!(msg = %msg, "agent said");
                Ok(())
            }) as TaskFuture
        }),
        agent_idle_callback: Arc::new(|| {
            Box::pin(async move {
                tracing::info!("agent idle");
                Ok(())
            }) as TaskFuture
        }),
    };

    // Spawn the runner.
    let run_handle = tokio::spawn(invocation.run(client));

    // Send initial user message.
    tx.send("Start counting from 0 to 5.".to_string())
        .await
        .unwrap();

    // Wait for either Stop signal or runner completion, then drop tx to close incoming.
    let mut run_handle = run_handle;
    tokio::select! {
        _ = stop_rx.recv() => {
            tracing::info!("stop signal received, dropping tx");
        }
        _ = &mut run_handle => {
            tracing::info!("run completed before stop signal");
        }
    }

    // Drop tx (and the clone) to close the incoming channel for run().
    drop(tx);

    run_handle
        .await
        .expect("agent loop should succeed")
        .expect("run ok");

    let final_count = *counter.lock().unwrap();
    assert!(
        final_count >= 5,
        "counter should have reached at least 5, got {}",
        final_count
    );
}
