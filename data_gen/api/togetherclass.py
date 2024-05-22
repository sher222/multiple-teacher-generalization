import threading
from openaiclass import OpenAIModel


class TogetherModel(OpenAIModel):
    BASEURL = "https://api.together.xyz"
    MAX_WORKERS = 256 # Maximum number of threads to use for sending requests
    RPI = 70  # Requests per interval limit
    INTERVAL = 1 # Interval in seconds to check the number of requests
    last_requests = []  # List to store timestamps of the last requests
    lock = threading.Lock()  # Lock to make checking the limit and sending requests thread-safe
    MODELS = [
        "meta-llama/Llama-3-70b-chat-hf"
    ]
    KEY_ENV_VAR = "TOGETHER_API_KEY" # TODO: change this depending on which key to use
    MAX_TOKENS = 600
if __name__ == "__main__":
    resp = TogetherModel("meta-llama/Llama-3-70b-chat-hf").get_responses([["hello"], ["hi"], ["hi"]], {"temperature": 0.9, "top_p": 1})
    print(resp)