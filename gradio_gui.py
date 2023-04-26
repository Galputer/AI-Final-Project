from general_cosine_sim_base import generate_top_ten
import gradio as gr


def gradio_gui():
    
    with gr.Blocks() as demo:
        keywords = gr.Textbox(label="Keywords (Seperate with a space. Do not leave a space after the last word)")
        semantics = gr.Radio(['both', 'pos', 'neg'], label="Do you want to see recommendations from only positive or negative reviews? (default is both)")
        years = gr.CheckboxGroup(["2000", "2001", "2002","2003", "2004", "2005"], label="Year", info="Recommendations based on specific review year(s)?")
        outputs = gr.Dataframe(row_count = (10, "dynamic"), col_count=(2, "dynamic"), label="Generated List")
        with gr.Tab("Baseline Cosine Similarity"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten, inputs = [keywords,semantics,years], outputs = outputs)
        with gr.Tab("Updated model"):
            generate_btn = gr.Button("generate")
            generate_btn.click(fn = generate_top_ten, inputs = [keywords,semantics,years], outputs = outputs)
    demo.queue()
    demo.launch(share=True)
    
    
if __name__ == "__main__":
    gradio_gui()