//! Shared runtime and protocol helpers for mesh-llm plugins.
//!
//! Plugins declare services in their manifest and implement typed service
//! handlers. MCP and HTTP are host-side projections over those services.

mod context;
mod dsl;
mod error;
mod helpers;
mod io;
mod manifest;
mod runtime;

pub use async_trait::async_trait;
pub use context::PluginContext;
pub use dsl::DeclarativePluginBuilder;
pub use error::{PluginError, PluginResult, PluginRpcResult, STARTUP_DISABLED_ERROR_CODE};
pub use helpers::{
    accept_bulk_transfer_message, bulk_transfer_message, bulk_transfer_sequence,
    cancel_task_result, channel_message, complete_result, empty_object_schema, get_prompt_result,
    get_task_payload_result, get_task_result, json_bytes, json_channel_message,
    json_reply_channel_message, json_response, json_schema_for, json_schema_operation, json_string,
    list_prompts, list_resource_templates, list_resources, list_tasks, list_tools, operation_error,
    operation_with_schema, parse_get_prompt_request, parse_optional_json,
    parse_read_resource_request, parse_rpc_params, plugin_server_info, plugin_server_info_full,
    prompt, prompt_argument, read_resource_result, resource_template, structured_tool_result, task,
    text_resource, BulkTransferSequence, CompletionFuture, CompletionRouter, JsonOperationFuture,
    OperationFuture, OperationRequest, OperationRouter, PromptFuture, PromptRouter, ResourceFuture,
    ResourceRouter, SubscriptionSet, TaskCancelFuture, TaskInfoFuture, TaskListFuture, TaskRecord,
    TaskResultFuture, TaskRouter, TaskStore,
};
pub use io::{
    bind_side_stream, connect_from_env, read_envelope, send_bulk_transfer_message,
    send_channel_message, write_envelope, LocalListener, LocalStream,
};
pub mod http {
    pub use crate::dsl::http::{delete, get, patch, post, put};
}
pub mod inference {
    pub use crate::dsl::inference::{openai_http, provider};
}
pub use manifest::{
    capability, completion, http_binding, http_delete, http_get, http_patch, http_post, http_put,
    mcp_http_endpoint, mcp_stdio_endpoint, mcp_tcp_endpoint, mcp_unix_socket_endpoint,
    openai_http_inference_endpoint, operation, plugin_manifest, prompt_service, resource,
    resource_template_service, CompletionBuilder, EndpointBuilder, HttpBindingBuilder,
    ManifestEntry, OperationBuilder, PluginManifestBuilder, PromptBuilder, ResourceBuilder,
    ResourceTemplateBuilder,
};
pub mod mcp {
    pub use crate::dsl::mcp::{
        completion, external_http, external_stdio, external_tcp, external_unix_socket, prompt,
        resource, resource_template, tool,
    };
}
pub use runtime::{
    MeshVisibility, Plugin, PluginInitializeRequest, PluginMetadata, PluginRuntime,
    PluginStartupPolicy, SimplePlugin,
};

#[allow(dead_code)]
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/meshllm.plugin.v1.rs"));
}

pub const PROTOCOL_VERSION: u32 = 1;

#[macro_export]
macro_rules! plugin_manifest {
    ($($item:expr),* $(,)?) => {{
        let mut builder = $crate::plugin_manifest();
        $(
            builder = builder.item($item);
        )*
        builder.build()
    }};
}

#[macro_export]
macro_rules! plugin {
    (
        metadata: $metadata:expr,
        $(provides: [$($provide:expr),* $(,)?],)?
        $(mcp: [$($mcp:expr),* $(,)?],)?
        $(http: [$($http:expr),* $(,)?],)?
        $(inference: [$($inference:expr),* $(,)?],)?
    ) => {{
        let mut builder = $crate::DeclarativePluginBuilder::new($metadata);
        $(
            $(
                builder = builder.provide($provide);
            )*
        )?
        $(
            $(
                builder = builder.mcp_item($mcp);
            )*
        )?
        $(
            $(
                builder = builder.http_item($http);
            )*
        )?
        $(
            $(
                builder = builder.inference_item($inference);
            )*
        )?
        builder.build()
    }};
}
