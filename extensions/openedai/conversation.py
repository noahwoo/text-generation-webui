import json
import os
import numpy as np
from termcolor import colored


paper_system_message = """You are arXivGPT, a helpful assistant pulls academic papers to answer user questions. 
You search for a list of papers to answer user questions. You read and summarize the paper clearly according to user provided topics.
Please summarize in clear and concise format. Begin!"""

class Conversation:
    def __init__(self, conversation_dir_filepath):
        self.conversation_history = []
        self.conversation_dir_filepath = conversation_dir_filepath

        self.system_setted = False
        if os.path.exists(self.conversation_dir_filepath):
            self.load_history()
            self.display_conversation()
        
        # no history, add system message
        if not self.system_setted :
            self.add_message("system", paper_system_message)

    def check_and_reset(self, question):
        conversation_list = question.split("\n")
        me_count = np.sum(1 for c in conversation_list if c.startswith("Me:"))
        if me_count > 1: 
            return
        self.conversation_history = self.conversation_history[:1]

    def add_message(self, role, content):
        message = {"role": role, "content": content}
        if role == "system":
            if self.system_setted == True:
                return
            self.system_setted = True
        
        # de-dup the last repeat question
        if len(self.conversation_history) > 0 and content == self.conversation_history[-1]["content"]:
            return
        
        # add message to history
        self.conversation_history.append(message)
        with open(self.conversation_dir_filepath, "w") as f:
            json.dump(self.conversation_history, f, ensure_ascii=False)

    def load_history(self) :
        with open(self.conversation_dir_filepath, "r") as f:
            self.conversation_history = json.load(f)
            for message in self.conversation_history :
                if message["role"] == "system" :
                    self.system_setted = True
                    break
        pass
    
    def display_conversation(self, detailed=False):
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )