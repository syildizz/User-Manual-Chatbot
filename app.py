
from gradio_app import gradio_main

if __name__ == "__main__":
    gr_interface = gradio_main()
    gr_interface.queue().launch()  # pyright: ignore[reportUnusedCallResult]