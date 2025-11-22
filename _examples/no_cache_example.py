from echollm.echollm import EchoLLM
from llm import Ollama
from llm.ollama_llm import OllamaModel


def run_no_cache_example():
    echo_llm = EchoLLM(
        cache=None,
        llm=Ollama(
            model=OllamaModel.GEMMA3_1B,
            host='http://localhost:11434',
        )
    )

    # First prompt must be a Miss
    request1 = 'Write me a short script of calculator in python'
    response1 = echo_llm.ask(request1)
    print(response1)
    print('-------------')

    # Suppose to be a Miss too although the exact question was just asked, due to not having cache at all
    request2 = 'Write me a short script of calculator in python'
    response2 = echo_llm.ask(request2)
    print(response2)


if __name__ == '__main__':
    run_no_cache_example()
