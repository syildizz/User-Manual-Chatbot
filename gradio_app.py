from typing import Any
import gradio as gr
from langchain_core.messages import AIMessage
from create_rag_agent import create_rag_agent

def gradio_main():

    rag_agent = create_rag_agent()

    def rag_agent_response(message: str, history: list[dict[str, Any]]):
        """
        The function integrated with Gradio, calling your LangChain rag_agent.
        It now passes the full conversation history for conversational context.
        
        The type hint for history is now the built-in generic: list[dict].
        """
        
        full_messages = history + [{"role": "user", "content": message}]

        agent_input = {
            "messages": full_messages
        }
        
        stream = rag_agent.stream(agent_input)

        current_response=""

        # Iterate over the stream of chunks
        for chunk in stream:

            model_in_chunk = chunk.get("model", [])

            if model_in_chunk:

                messages_in_chunk = model_in_chunk.get("messages", [])
                
                if messages_in_chunk:
                    # The final item in the messages list contains the generated text chunk
                    message_chunk = messages_in_chunk[-1]
                    
                    # We use getattr to safely get the content from a message object/chunk
                    content_chunk = getattr(message_chunk, "text", None)
                    
                    if content_chunk:
                        # Accumulate and yield the running response
                        current_response += content_chunk
                        yield current_response

    gr_interface = gr.ChatInterface(
        fn=rag_agent_response, 
        type="messages",
        chatbot=gr.Chatbot(
            height=500, 
            label="User Manual Chatbot",
            type="messages"
        ),
        textbox=gr.Textbox(placeholder="Enter your query here...", container=False, scale=7),
        title="User Manual Chatbot",
        description="Ask any technical question you wish",
        theme="soft"
    )

    return gr_interface

if __name__ == "__main__":
    gradio_main().queue().launch()  # pyright: ignore[reportUnusedCallResult]