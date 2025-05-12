import os
import copy
from google.cloud import aiplatform

# Path to the service account key JSON file
service_account_key_path = './access_forrestor_nlp.json'

# Set the environment variable GOOGLE_APPLICATION_CREDENTIALS to the service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = service_account_key_path

###
# For Tool calling refer
# https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling
###

import base64
import vertexai
import time
from vertexai.generative_models import GenerativeModel, SafetySetting, FinishReason, Tool, ToolConfig
import vertexai.generative_models as generative_models

vertexai.init(project="forrester-nlp", location="us-central1")

payload = {
    "max_output_tokens": 8192,
    "temperature": 0.2,
    "top_p": 0.95,
}

safety_settings = {
    SafetySetting.HarmCategory.HARM_CATEGORY_HATE_SPEECH: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    SafetySetting.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    SafetySetting.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH,
    SafetySetting.HarmCategory.HARM_CATEGORY_HARASSMENT: SafetySetting.HarmBlockThreshold.BLOCK_ONLY_HIGH
}


def get_response(input_messages, audio_path=None, return_json=False, function_declarations=None,
                 model="gemini-2.0-flash", system_message=None, **kwargs):
    temp_payload = copy.deepcopy(payload)
    temp_payload.update(kwargs)
    if return_json:
        temp_payload["response_mime_type"] = "application/json"

    if system_message is not None:
        model = GenerativeModel(
            model,
            system_instruction=[system_message]
        )
    else:
        model = GenerativeModel(
            model
        )

    try:
        if function_declarations is None:
            response = model.generate_content(
                input_messages,
                generation_config=temp_payload,
                safety_settings=safety_settings,
            )
        else:
            tool = Tool(
                function_declarations=function_declarations,
            )
            response = model.generate_content(
                input_messages,
                generation_config=temp_payload,
                tools=[tool],
                tool_config=ToolConfig(
                    function_calling_config=ToolConfig.FunctionCallingConfig(
                        mode=ToolConfig.FunctionCallingConfig.Mode.AUTO,
                    )
                )
            )
        return response
    except Exception as e:
        print(f"GEMINI CALL ERROR {e}")
        return None

