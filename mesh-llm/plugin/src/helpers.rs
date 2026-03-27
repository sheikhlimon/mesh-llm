use rmcp::model::{
    CallToolResult, Content, GetPromptRequestParams, GetPromptResult, Implementation,
    ListPromptsResult, ListResourceTemplatesResult, ListResourcesResult, ListTasksResult,
    ListToolsResult, Prompt, PromptArgument, ReadResourceRequestParams, ReadResourceResult,
    Resource, ResourceContents, ResourceTemplate, ServerCapabilities, ServerInfo, Task, TaskStatus,
    Tool, CompleteResult, CompletionInfo, GetTaskPayloadResult, GetTaskResult, CancelTaskResult,
};
use rmcp::model::{AnnotateAble, RawResource, RawResourceTemplate};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde::de::DeserializeOwned;
use serde_json::json;
use std::sync::Arc;

use crate::{
    error::{PluginError, PluginResult, PluginRpcResult},
    proto,
};

fn default_arguments() -> serde_json::Value {
    json!({})
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ToolCallRequest {
    pub name: String,
    #[serde(default = "default_arguments")]
    pub arguments: serde_json::Value,
}

impl ToolCallRequest {
    pub fn arguments<T: DeserializeOwned>(&self) -> PluginResult<T> {
        serde_json::from_value(self.arguments.clone()).map_err(|err| {
            PluginError::invalid_params(format!(
                "Invalid arguments for tool '{}': {err}",
                self.name
            ))
        })
    }

    pub fn arguments_or_default<T>(&self) -> PluginResult<T>
    where
        T: DeserializeOwned + Default,
    {
        self.arguments()
    }
}

pub fn json_string<T: Serialize>(value: &T) -> PluginResult<String> {
    serde_json::to_string(value).map_err(|err| PluginError::internal(err.to_string()))
}

pub fn json_bytes<T: Serialize>(value: &T) -> PluginResult<Vec<u8>> {
    serde_json::to_vec(value).map_err(|err| PluginError::internal(err.to_string()))
}

pub fn structured_tool_result<T: Serialize>(value: T) -> PluginResult<CallToolResult> {
    let value = serde_json::to_value(value).map_err(|err| PluginError::internal(err.to_string()))?;
    Ok(CallToolResult::structured(value))
}

pub fn tool_error(message: impl Into<String>) -> CallToolResult {
    CallToolResult::error(vec![Content::text(message.into())])
}

pub fn list_tools(tools: Vec<Tool>) -> ListToolsResult {
    ListToolsResult {
        tools,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_prompts(prompts: Vec<Prompt>) -> ListPromptsResult {
    ListPromptsResult {
        prompts,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_resources(resources: Vec<Resource>) -> ListResourcesResult {
    ListResourcesResult {
        resources,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_resource_templates(resource_templates: Vec<ResourceTemplate>) -> ListResourceTemplatesResult {
    ListResourceTemplatesResult {
        resource_templates,
        meta: None,
        next_cursor: None,
    }
}

pub fn list_tasks(tasks: Vec<Task>) -> ListTasksResult {
    ListTasksResult::new(tasks)
}

pub fn read_resource_result(contents: Vec<ResourceContents>) -> ReadResourceResult {
    ReadResourceResult::new(contents)
}

pub fn get_prompt_result(messages: Vec<rmcp::model::PromptMessage>) -> GetPromptResult {
    GetPromptResult::new(messages)
}

pub fn complete_result(values: Vec<String>) -> PluginResult<CompleteResult> {
    let completion = CompletionInfo::with_all_values(values)
        .map_err(PluginError::invalid_params)?;
    Ok(CompleteResult::new(completion))
}

pub fn prompt(
    name: impl Into<String>,
    description: impl Into<String>,
    arguments: Option<Vec<PromptArgument>>,
) -> Prompt {
    Prompt::new(name, Some(description.into()), arguments)
}

pub fn prompt_argument(
    name: impl Into<String>,
    description: impl Into<String>,
    required: bool,
) -> PromptArgument {
    PromptArgument::new(name)
        .with_description(description)
        .with_required(required)
}

pub fn text_resource(uri: impl Into<String>, name: impl Into<String>) -> Resource {
    RawResource::new(uri, name).no_annotation()
}

pub fn resource_template(uri_template: impl Into<String>, name: impl Into<String>) -> ResourceTemplate {
    RawResourceTemplate::new(uri_template, name).no_annotation()
}

pub fn task(
    task_id: impl Into<String>,
    status: TaskStatus,
    created_at: impl Into<String>,
    last_updated_at: impl Into<String>,
) -> Task {
    Task::new(task_id.into(), status, created_at.into(), last_updated_at.into())
}

pub fn get_task_result(task: Task) -> GetTaskResult {
    GetTaskResult { meta: None, task }
}

pub fn get_task_payload_result<T: Serialize>(value: T) -> PluginResult<GetTaskPayloadResult> {
    let value = serde_json::to_value(value).map_err(|err| PluginError::internal(err.to_string()))?;
    Ok(GetTaskPayloadResult::new(value))
}

pub fn cancel_task_result(task: Task) -> CancelTaskResult {
    CancelTaskResult { meta: None, task }
}

pub fn plugin_server_info(
    implementation_name: impl Into<String>,
    implementation_version: impl Into<String>,
    title: impl Into<String>,
    description: impl Into<String>,
    instructions: Option<impl Into<String>>,
) -> ServerInfo {
    let info = ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
        .with_server_info(
            Implementation::new(implementation_name, implementation_version)
                .with_title(title)
                .with_description(description),
        );
    match instructions {
        Some(instructions) => info.with_instructions(instructions.into()),
        None => info,
    }
}

pub fn empty_object_schema() -> serde_json::Map<String, serde_json::Value> {
    serde_json::json!({
        "type": "object",
        "additionalProperties": false
    })
    .as_object()
    .cloned()
    .unwrap()
}

pub fn json_schema_for<T: JsonSchema>() -> serde_json::Map<String, serde_json::Value> {
    serde_json::to_value(schemars::schema_for!(T))
        .ok()
        .and_then(|value| value.as_object().cloned())
        .unwrap_or_else(|| {
            serde_json::json!({
                "type": "object",
                "additionalProperties": true
            })
            .as_object()
            .cloned()
            .unwrap()
        })
}

pub fn tool_with_schema(
    name: impl Into<String>,
    description: impl Into<String>,
    schema: serde_json::Map<String, serde_json::Value>,
) -> Tool {
    Tool::new(name.into(), description.into(), Arc::new(schema))
}

pub fn json_schema_tool<T: JsonSchema>(
    name: impl Into<String>,
    description: impl Into<String>,
) -> Tool {
    tool_with_schema(name, description, json_schema_for::<T>())
}

pub fn channel_message(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    body: Vec<u8>,
    message_kind: impl Into<String>,
) -> proto::ChannelMessage {
    proto::ChannelMessage {
        channel: channel.into(),
        source_peer_id: String::new(),
        target_peer_id: target_peer_id.into(),
        content_type: content_type.into(),
        body,
        message_kind: message_kind.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
    }
}

pub fn json_channel_message<T: Serialize>(
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    message_kind: impl Into<String>,
    payload: &T,
) -> PluginResult<proto::ChannelMessage> {
    Ok(channel_message(
        channel,
        target_peer_id,
        "application/json",
        json_bytes(payload)?,
        message_kind,
    ))
}

pub fn bulk_transfer_message(
    kind: i32,
    channel: impl Into<String>,
    target_peer_id: impl Into<String>,
    content_type: impl Into<String>,
    total_bytes: u64,
    offset: u64,
    body: Vec<u8>,
    final_chunk: bool,
) -> proto::BulkTransferMessage {
    proto::BulkTransferMessage {
        kind,
        transfer_id: String::new(),
        channel: channel.into(),
        source_peer_id: String::new(),
        target_peer_id: target_peer_id.into(),
        content_type: content_type.into(),
        correlation_id: String::new(),
        metadata_json: String::new(),
        total_bytes,
        offset,
        body,
        final_chunk,
    }
}

pub fn json_response<T: Serialize>(value: &T) -> PluginRpcResult {
    Ok(proto::envelope::Payload::RpcResponse(proto::RpcResponse {
        result_json: serde_json::to_string(value)
            .map_err(|err| PluginError::internal(err.to_string()))?,
    }))
}

pub fn parse_rpc_params<T: DeserializeOwned>(
    request: &proto::RpcRequest,
) -> Result<T, PluginError> {
    serde_json::from_str(&request.params_json).map_err(|err| {
        PluginError::invalid_params(format!("Invalid params for '{}': {err}", request.method))
    })
}

pub fn parse_tool_call_request(request: &proto::RpcRequest) -> PluginResult<ToolCallRequest> {
    parse_rpc_params(request)
}

pub fn parse_optional_json(raw: &str) -> Option<serde_json::Value> {
    if raw.trim().is_empty() {
        None
    } else {
        serde_json::from_str(raw).ok()
    }
}

pub fn parse_get_prompt_request(
    request: &proto::RpcRequest,
) -> PluginResult<GetPromptRequestParams> {
    parse_rpc_params(request)
}

pub fn parse_read_resource_request(
    request: &proto::RpcRequest,
) -> PluginResult<ReadResourceRequestParams> {
    parse_rpc_params(request)
}
