import os
import json
import asyncio
from typing import List, Tuple, Dict, Any, Optional, Union

import vertexai
from vertexai.generative_models import (
    GenerativeModel,
    FunctionDeclaration,
    Tool,
    ToolConfig,
    Content,
    Part,
)


class VertexAIHelper:
    """
    Tiny helper around the Vertex-AI Generative SDK that can be used
    as a drop-in replacement for “OpenAI-style” function calling code.
    """

    def __init__(self, service_account_key: str, location: str = "us-central1"):
        if not os.path.exists(service_account_key):
            raise FileNotFoundError(service_account_key)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key
        with open(service_account_key, "r", encoding="utf-8") as f:
            project_id = json.load(f).get("project_id")
        if not project_id:
            raise ValueError("Could not find `project_id` inside the key file.")
        vertexai.init(project=project_id, location=location)

        self._generation_defaults = {
            "max_output_tokens": 1024,
            "temperature": 0.2,
            "top_p": 0.95,
            # you can override these by passing max_retries or backoff_base in generation_overrides
            "max_retries": 3,
            "backoff_base": 1,
        }

    async def chat(
        self,
        messages: Union[List[Tuple[str, str]], str],
        model_name: str = "gemini-2.0-flash",
        tools: Optional[List[Dict[str, Any]]] = None,
        system_message: Optional[str] = None,
        generation_overrides: Optional[Dict[str, Any]] = None,
        **extra_generation_kwargs,
    ) -> List[str]:
        # direct single-call path; returns either List[str] or str
        result = await self._chat_single_async(
            model_name=model_name,
            messages=messages,
            system_message=system_message,
            tools=tools,
            generation_overrides=generation_overrides,
            **extra_generation_kwargs,
        )
        return result

    #TODO: Put a cap on the maximum number of concurrent requests too like in aqm predict
    async def chat_batch(
            self,
            all_messages: List[List[Tuple[str, str]]],
            model_name: str = "gemini-2.0-flash",
            tools: Optional[
                Union[
                    List[Dict[str, Any]],  # one global list
                    List[List[Dict[str, Any]]],  # per-conversation lists
                ]
            ] = None,
            system_message: Optional[
                Union[
                    str,  # one global
                    List[str],  # one per-conversation
                ]
            ] = None,
            generation_overrides: Optional[Dict[str, Any]] = None,
            rps: float = 1.0,
            **extra_generation_kwargs,
    ) -> List[List[str]]:
        if rps <= 0:
            raise ValueError("`rps` must be a positive number.")
        interval = 1.0 / rps

        tasks = []

        async def worker(
                msgs: List[Tuple[str, str]],
                sys_msg: Optional[str],
                tools_list: Optional[List[Dict[str, Any]]],
        ) -> List[str]:
            return await self._chat_single_async(
                model_name=model_name,
                messages=msgs,
                system_message=sys_msg,
                tools=tools_list,
                generation_overrides=generation_overrides,
                **extra_generation_kwargs,
            )

        for i, msgs in enumerate(all_messages):
            # per-conversation system message
            if isinstance(system_message, list):
                sys_i = system_message[i]
            else:
                sys_i = system_message

            # per-conversation tools
            if not tools:
                tools_i = None
            elif isinstance(tools[0], dict):
                tools_i = tools
            else:
                tools_i = tools[i]

            # schedule the call
            tasks.append(asyncio.create_task(worker(msgs, sys_i, tools_i)))
            # throttle by RPS
            await asyncio.sleep(interval)

        return await asyncio.gather(*tasks)

    async def _chat_single_async(
        self,
        model_name: str,
        messages:  Union[List[Tuple[str, str]], str],
        system_message: Optional[str],
        tools: Optional[List[Dict[str, Any]]],
        generation_overrides: Optional[Dict[str, Any]],
        **extra_generation_kwargs,
    ) -> Union[List[str], str]:
        # 1) build model handle
        if system_message:
            model = GenerativeModel(model_name, system_instruction=[system_message])
        else:
            model = GenerativeModel(model_name)

        # 2) translate tools
        vertex_fns = self._openai_tools_to_vertex(tools)
        tool_obj = Tool(function_declarations=vertex_fns) if vertex_fns else None

        # 3) merge generation config
        gen_cfg = {
            **self._generation_defaults,
            **(generation_overrides or {}),
            **extra_generation_kwargs,
        }
        max_retries = gen_cfg.pop("max_retries", 3)
        backoff_base = gen_cfg.pop("backoff_base", 1)

        if isinstance(messages, str):
            messages = [('user', messages)]

        # 4) build the Content list
        vertex_messages = [
            Content(role=role, parts=[Part.from_text(text)])
            for role, text in messages
        ]

        # 5) call with retry on rate-limit / 429
        last_exc: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                if tool_obj:
                    resp = await model.generate_content_async(
                        vertex_messages,
                        generation_config=gen_cfg,
                        tools=[tool_obj],
                        tool_config=ToolConfig(
                            function_calling_config=ToolConfig.FunctionCallingConfig(
                                mode=ToolConfig.FunctionCallingConfig.Mode.AUTO
                            )
                        ),
                    )
                else:
                    resp = await model.generate_content_async(
                        vertex_messages,
                        generation_config=gen_cfg,
                    )
                break
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                # only retry on rate-limit / 429 clues
                if ("rate limit" in msg or "limit exceeded" in msg or "429" in msg) and attempt < max_retries - 1:
                    await asyncio.sleep(backoff_base * (2 ** attempt))
                    continue
                # otherwise fail fast
                raise
        else:
            # all retries exhausted
            raise last_exc  # type: ignore

        # 6) unpack
        try:
            parts = resp.to_dict()["candidates"][0]["content"]["parts"]
            processed_parts = [p.get("text", p.get("function_call")) for p in parts]
            if len(processed_parts) == 1 and isinstance(processed_parts[0], str):
                return processed_parts[0]
            #In case the model returns a mix of both text and function calls, return only the function calls
            #ignoring the text
            return [p for p in processed_parts if not isinstance(p, str)]
        except Exception:
            # fallback: return finish_reason
            cand = resp.to_dict()["candidates"][0]
            return [cand.get("finish_reason", "unknown")]

    @staticmethod
    def _openai_tools_to_vertex(
        tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[FunctionDeclaration]]:
        if not tools:
            return None

        vertex_fns: List[FunctionDeclaration] = []
        for tool in tools:
            if tool.get("type") != "function" or "function" not in tool:
                continue
            fn = tool["function"]
            params = fn.get("parameters") or {}
            if isinstance(params, dict):
                params = {k: v for k, v in params.items() if k != "additionalProperties"}
            vertex_fns.append(
                FunctionDeclaration(
                    name=fn.get("name"),
                    description=fn.get("description", ""),
                    parameters=params,
                )
            )
        return vertex_fns or None