import base64

import llama_cpp.llama_chat_format
import Llava15ChatHandler
from llama_cpp import Llama


def convert_bytes

chat_handler = Llava15ChatHandler(clip_model_path="path/to/llava/mmproj.bin")
llm = Llama(
    model_path="./path/to/llava/llama-model.gguf",
    chat_handler=chat_handler,
    n_ctx=2048,  # n_ctx should be increased to accomodate the image embedding
)

llm.create_chat_completion(
        messages = [
            {"role":"system","content":"you're an assistant who perfectly describes images"},
            {"role":"user",
             "content":[
                 {"type":"image_url", "image_url": {"url":"https://../image.png"}}
                 {"type":"text","text":"Describe this image in detail please"}
                 ]
             }
            ]
        )
