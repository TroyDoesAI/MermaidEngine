import transformers
import torch
import json

class Mermaid:
    def __init__(self, model_id):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device_map="auto",
        )
    
    def generate_markdown_template(self, context, input_text, instruction):
        template = f"""Contextual-Request:
BEGININPUT
BEGINCONTEXT
{context}
ENDCONTEXT
{input_text}
ENDINPUT
BEGININSTRUCTION
{instruction}
ENDINSTRUCTION

### Contextual Response:
"""
        return template

    def generate_response(self, template):
        outputs = self.pipeline(
            template,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )
        
        response = outputs[0]["generated_text"].strip()
        return response

def main():
    # Load model ID from config file
    with open('config.json', 'r') as f:
        config = json.load(f)
        model_id = config.get('model_id')

    mermaid = Mermaid(model_id)

    # Example usage
    context = "date: {_DATE}\nurl: {_URL}"
    input_text = "Pandemic Warning Notice there has been a huge issue with Zombie humans that are passing on a new disease that appeared to be similar to the symptoms of covid but when a host dies they reanimate as a zombie corpse. It also turns gay people straight."
    instruction = "What is the pandemic about? Cite your sources."

    template = mermaid.generate_markdown_template(context, input_text, instruction)
    print("Markdown Template:")
    print(template)

    response = mermaid.generate_response(template)
    print("Generated Response:", response)

if __name__ == "__main__":
    main()
