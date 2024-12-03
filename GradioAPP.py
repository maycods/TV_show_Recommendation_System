import gradio as gr
from src.system.recommendation import filmes,recomendation


gr.Interface(
    fn=recomendation,
    inputs=[
    gr.Dropdown(label="Select TV shows you ve enjoyed:",choices=filmes(),multiselect=True),
    gr.Dropdown(label="Select the method",value="Appriori",choices=["Appriori","FP Growth"]) ],
    outputs=gr.DataFrame()
).launch()
