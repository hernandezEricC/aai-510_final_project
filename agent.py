import json
from typing import Any, Callable, Generator, Optional
from uuid import uuid4
import warnings

import backoff
import mlflow
import openai
from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent
from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
    create_function_call_output_item,
    create_text_output_item,
)
from openai import OpenAI
from pydantic import BaseModel
from unitycatalog.ai.core.base import get_uc_function_client

############################################
# Define your LLM endpoint and system prompt
############################################
LLM_ENDPOINT_NAME = "__LLM_ENDPOINT__"

SYSTEM_PROMPT = """
You are a restaurant comparison assistant focused on San Diego Italian restaurants.

Your scope:
- Answer questions about San Diego Italian restaurants using the available restaurant metrics and review-text tools.
- Help users compare restaurants, understand ratings, review counts, rating distributions, and customer review themes.
- Provide recommendations only when they can be grounded in the available tool outputs.

Out-of-scope requests:
- Do not answer questions unrelated to San Diego Italian restaurants.
- Do not provide general homework help, coding help, medical advice, legal advice, financial advice, travel planning, politics, sports, entertainment, or unrelated factual information.
- Do not invent restaurant data, restaurant names, metrics, ratings, review counts, review text, or business recommendations.

If the user asks an out-of-scope question, politely reject it and briefly explain that you can only help with San Diego Italian restaurant comparison questions using the available data.

You have access to three tools:
1. business_lookup: retrieves restaurant-level metrics such as average rating, review count, and rating distribution.
2. competitor_comparison: compares two restaurants using rating, review volume, and rating distribution.
3. theme_retrieval: retrieves semantically relevant review excerpts for qualitative customer-experience themes.

Tool-use rules:
- For structured questions about one restaurant, use business_lookup.
- For structured comparison questions about two restaurants, use competitor_comparison.
- For qualitative questions about customer experiences, use theme_retrieval.
- For broad comparison, recommendation, or “what does one restaurant do better” questions, use both competitor_comparison and theme_retrieval. The structured comparison shows rating and review-count differences, while theme_retrieval provides specific evidence about what customers mention, such as service, food quality, ambiance, authenticity, value, wait time, and special occasions.
- If a tool returns no results, say that the data was not found.

Use the restaurant names provided by the user. Do not assume the user only cares about a fixed pair of restaurants.
Be concise, but include enough evidence from tool outputs to justify the answer.
"""

###############################################################################
## Define tools for your agent, enabling it to retrieve data or take actions
## beyond text generation
## To create and see usage examples of more tools, see
## https://docs.databricks.com/generative-ai/agent-framework/agent-tool.html
###############################################################################
class ToolInfo(BaseModel):
    """
    Class representing a tool for the agent.
    - "name" (str): The name of the tool.
    - "spec" (dict): JSON description of the tool (matches OpenAI Responses format)
    - "exec_fn" (Callable): Function that implements the tool logic
    """

    name: str
    spec: dict
    exec_fn: Callable

def create_tool_info(tool_spec, exec_fn_param: Optional[Callable] = None):
    tool_spec["function"].pop("strict", None)
    tool_name = tool_spec["function"]["name"]
    udf_name = tool_name.replace("__", ".")

    # Define a wrapper that accepts kwargs for the UC tool call,
    # then passes them to the UC tool execution client
    def exec_fn(**kwargs):
        function_result = uc_function_client.execute_function(udf_name, kwargs)
        if function_result.error is not None:
            return function_result.error
        else:
            return function_result.value
    return ToolInfo(name=tool_name, spec=tool_spec, exec_fn=exec_fn_param or exec_fn)

TOOL_INFOS = []

# You can use UDFs in Unity Catalog as agent tools
UC_TOOL_NAMES = ["main.aai510_final_agent.business_lookup", 
                 "main.aai510_final_agent.competitor_comparison",
                 "main.aai510_final_agent.theme_retrieval"]

uc_toolkit = UCFunctionToolkit(function_names=UC_TOOL_NAMES)
uc_function_client = get_uc_function_client()
for tool_spec in uc_toolkit.tools:
    TOOL_INFOS.append(create_tool_info(tool_spec))

def _sanitize_tool_spec(spec: dict) -> dict:
    """Remove JSON schema keywords that some model endpoints reject (e.g. minLength, minimum, minItems, format)."""
    import copy
    
    def sanitize_schema(schema):
        """Recursively sanitize a JSON schema object."""
        if not isinstance(schema, dict):
            return
        
        t = schema.get("type")
        if t == "string":
            for key in ("minLength", "maxLength", "pattern", "format"):
                schema.pop(key, None)
        elif t in ("integer", "number"):
            for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
                schema.pop(key, None)
        elif t == "array":
            for key in ("minItems", "maxItems", "uniqueItems"):
                schema.pop(key, None)
            if "items" in schema:
                sanitize_schema(schema["items"])
        
        if "properties" in schema:
            for prop in schema["properties"].values():
                sanitize_schema(prop)
        
        if "additionalProperties" in schema and isinstance(schema["additionalProperties"], dict):
            sanitize_schema(schema["additionalProperties"])
    
    spec = copy.deepcopy(spec)
    params = spec.get("function", {}).get("parameters") or {}
    if isinstance(params, dict):
        sanitize_schema(params)
    return spec


class RestaurantToolCallingAgent:
    """
    Class representing a tool-calling Agent
    """

    def __init__(self, llm_endpoint: str, tools: list[ToolInfo]):
        """Initializes the ToolCallingAgent with tools."""
        self.llm_endpoint = llm_endpoint
        self.workspace_client = WorkspaceClient()
        self.model_serving_client: OpenAI = (
            self.workspace_client.serving_endpoints.get_open_ai_client()
        )
        self._tools_dict = {tool.name: tool for tool in tools}

    def get_tool_specs(self) -> list[dict]:
        """Returns tool specifications in the format OpenAI expects. Strips schema keywords (e.g. minLength) that some endpoints reject."""
        return [_sanitize_tool_spec(tool_info.spec) for tool_info in self._tools_dict.values()]

    @mlflow.trace(span_type=SpanType.TOOL)
    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Executes the specified tool with the given arguments."""
        # UC rejects params like {'': '{}'} for no-arg functions; keep only valid keyed args
        sane_args = {k: v for k, v in (args or {}).items() if k and isinstance(k, str)}
        # Normalize: strip quotes and model tokens (e.g. <|channel|>commentary) appended to name
        name = tool_name.strip().strip('"').strip("'")
        if "<" in name:
            name = name.split("<")[0].strip()
        if name in self._tools_dict:
            return self._tools_dict[name].exec_fn(**sane_args)
        # Fallback: name may have extra tokens; use longest key that name starts with
        candidates = [k for k in self._tools_dict if name.startswith(k)]
        if candidates:
            return self._tools_dict[max(candidates, key=len)].exec_fn(**sane_args)
        raise KeyError(f"Unknown tool: {tool_name!r}. Known tools: {list(self._tools_dict.keys())}")

    def call_llm(self, messages: list[dict[str, Any]]) -> Generator[dict[str, Any], None, None]:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="PydanticSerializationUnexpectedValue")
            for chunk in self.model_serving_client.chat.completions.create(
                model=self.llm_endpoint,
                messages=to_chat_completions_input(messages),
                tools=self.get_tool_specs(),
                stream=True,
            ):
                chunk_dict = chunk.to_dict()
                if len(chunk_dict.get("choices", [])) > 0:
                    yield chunk_dict

    def handle_tool_call(
        self,
        tool_call: dict[str, Any],
        messages: list[dict[str, Any]],
    ) -> ResponsesAgentStreamEvent:
        """
        Execute tool calls, add them to the running message history, and return a ResponsesStreamEvent w/ tool output
        """
        try:
            args = json.loads(tool_call.get("arguments"))
        except Exception as e:
            args = {}
        result = str(self.execute_tool(tool_name=tool_call["name"], args=args))

        tool_call_output = create_function_call_output_item(tool_call["call_id"], result)
        messages.append(tool_call_output)
        return ResponsesAgentStreamEvent(type="response.output_item.done", item=tool_call_output)

    def call_and_run_tools(
        self,
        messages: list[dict[str, Any]],
        max_iter: int = 20,
    ) -> Generator[ResponsesAgentStreamEvent, None, None]:
        for _ in range(max_iter):
            last_msg = messages[-1]
            if last_msg.get("role", None) == "assistant":
                return
            elif last_msg.get("type", None) == "function_call":
                yield self.handle_tool_call(last_msg, messages)
            else:
                yield from output_to_responses_items_stream(
                    chunks=self.call_llm(messages), aggregator=messages
                )

        yield ResponsesAgentStreamEvent(
            type="response.output_item.done",
            item=create_text_output_item("Max iterations reached. Stopping.", str(uuid4())),
        )

    def predict(self, request: ResponsesAgentRequest) -> ResponsesAgentResponse:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(output=outputs, custom_outputs=request.custom_inputs)

    def predict_stream(self, request: ResponsesAgentRequest) -> Generator[ResponsesAgentStreamEvent, None, None]:
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        messages = to_chat_completions_input([i.model_dump() for i in request.input])
        if SYSTEM_PROMPT:
            messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT})
        yield from self.call_and_run_tools(messages=messages)


# Log the model using MLflow (AGENT is created in the notebook after adding MCP tools)
mlflow.openai.autolog()
