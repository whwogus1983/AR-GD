import gradio

def greet(name):
    return 'hi ' + name

app = gradio.Interface(
    inputs="text",
    fn=greet,
    outputs="text"
)

app.launch(share=True)

'''
import gradio as gr

def display_value(num):
    return num

iface = gr.Interface(
    fn=display_value, 
    inputs="number", 
    outputs=gr.outputs.Textbox(), 
    title="Number Display",
    description="Displays a number as a moving bar",
    theme="red", 
    layout="horizontal", 
    live=True,
    bar_color="#FF0000",
    granularity=1,
    inputs_label="Number Input",
    outputs_label="Value Display"
)

iface.launch(inline=False)

while True:
    num = int(input("Enter a number between 1 and 100: "))
    if num < 1 or num > 100:
        print("Invalid input! Number must be between 1 and 100.")
        continue
    iface.set_inputs(num)
위 코드를 실행하면 gradio의 인터페이스가 열리고, 사용자가 입력한 값에 따라 작은 바가 움직입니다. 사용자가 입력한 값이 1에서 100 사이의 범위를 벗어나면, "Invalid input! Number must be between 1 and 100." 메시지가 표시됩니다.
'''
