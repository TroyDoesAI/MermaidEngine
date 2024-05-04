import sys
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QTextEdit, QVBoxLayout, QWidget, QStackedWidget, QFrame, QHBoxLayout, QPushButton

class Mermaid:
    def __init__(self, model_id):
        import transformers
        import torch
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.float16},
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

class MermaidGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.config = self.load_config()
        self.mermaid = Mermaid(self.config['model_id']) if self.config.get('model_id') else None
        self.initUI()

    def load_config(self):
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("Config file not found.")
            return {'model_id': ''}

    def initUI(self):
        self.setWindowTitle('Mermaid GUI')
        self.setGeometry(100, 100, 800, 600)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout()
        main_widget.setLayout(layout)

        self.model_label = QLabel('Selected Model: ' + (self.config['model_id'] if self.config['model_id'] else 'None'))
        layout.addWidget(self.model_label)

        # Main Content Area
        self.stacked_widget = QStackedWidget()
        layout.addWidget(self.stacked_widget)

        # Contextual Page
        self.contextual_page = self.create_contextual_page()
        self.stacked_widget.addWidget(self.contextual_page)

        # Mermaid Flow Page
        self.mermaidflow_page = self.create_simple_page("Input your text and generate a response:")
        self.stacked_widget.addWidget(self.mermaidflow_page)

        # Page Navigation Buttons
        page_nav_frame = QFrame()
        page_nav_layout = QHBoxLayout()
        page_nav_frame.setLayout(page_nav_layout)

        contextual_button = QPushButton('Contextual')
        contextual_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        page_nav_layout.addWidget(contextual_button)

        mermaidflow_button = QPushButton('Mermaid Flow')
        mermaidflow_button.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        page_nav_layout.addWidget(mermaidflow_button)

        layout.addWidget(page_nav_frame)

    def create_contextual_page(self):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        context_edit = QTextEdit()
        context_edit.setPlaceholderText("Context: date: {_DATE}\nurl: {_URL}")
        layout.addWidget(context_edit)

        input_edit = QTextEdit()
        input_edit.setPlaceholderText("Input Text: Pandemic Warning Notice...")
        layout.addWidget(input_edit)

        instruction_edit = QTextEdit()
        instruction_edit.setPlaceholderText("Instruction: What is the pandemic about? Cite your sources.")
        layout.addWidget(instruction_edit)

        response_label = QLabel('Response:')
        layout.addWidget(response_label)

        generate_button = QPushButton('Generate Response')
        generate_button.clicked.connect(lambda: self.generate_contextual_response([context_edit, input_edit, instruction_edit], response_label))
        layout.addWidget(generate_button)

        return page

    def create_simple_page(self, placeholder):
        page = QWidget()
        layout = QVBoxLayout()
        page.setLayout(layout)

        text_edit = QTextEdit()
        text_edit.setPlaceholderText(placeholder)
        layout.addWidget(text_edit)

        response_label = QLabel('Response:')
        layout.addWidget(response_label)

        generate_button = QPushButton('Generate')
        generate_button.clicked.connect(lambda: self.generate_simple_response(text_edit.toPlainText(), response_label))
        layout.addWidget(generate_button)

        return page

    def generate_contextual_response(self, edits, response_label):
        if self.mermaid:
            context = edits[0].toPlainText()
            input_text = edits[1].toPlainText()
            instruction = edits[2].toPlainText()
            template = self.mermaid.generate_markdown_template(context, input_text, instruction)
            response = self.mermaid.generate_response(template)
            response_label.setText(f'Response: {response}')
        else:
            response_label.setText("No model loaded. Please configure the model in config.json.")

    def generate_simple_response(self, input_text, response_label):
        if self.mermaid:
            response = self.mermaid.generate_response(input_text)
            response_label.setText(f'Response: {response}')
        else:
            response_label.setText("No model loaded. Please configure the model in config.json.")

def main():
    app = QApplication(sys.argv)
    gui = MermaidGUI()
    gui.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
