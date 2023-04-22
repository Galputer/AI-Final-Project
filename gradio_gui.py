from cosine_similarity_baseline import generate_top_ten
import gradio as gr


def gradio_gui():
    
    with gr.Blocks() as demo:
        with gr.Tab("Baseline Cosine Similarity"):
            keywords = gr.Textbox(label="Keywords")
            generate_btn = gr.Button("generate")
            outputs = gr.Dataframe(row_count = (10, "dynamic"), col_count=(2, "dynamic"), label="Generated List")
            
            generate_btn.click(fn = generate_top_ten, inputs = keywords, outputs = outputs)
        with gr.Tab("Updated model"):
            keywords_2 = gr.Textbox(label="Keywords")
            generate_btn_2 = gr.Button("generate")
            outputs_2 = gr.Dataframe(row_count = (10, "dynamic"), col_count=(2, "dynamic"), label="Generated List")
            
            # TODO 
            generate_btn_2.click(fn = generate_top_ten, inputs = keywords_2, outputs = outputs_2)
    
    demo.queue()
    demo.launch(share=True)
    
    
if __name__ == "__main__":
    gradio_gui()