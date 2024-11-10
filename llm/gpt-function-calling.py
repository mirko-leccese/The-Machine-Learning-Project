###### GPT FUNCTION CALLING ########
# Python script to illustrate the use of function calling with OpenAI GPT models. Specifically, we illustrate a simple example
# where we instruct GPT to use custom python functions to compute basic statistics on a provided list of numbers. 

from openai import OpenAI
import numpy as np
import json

# Read openai-conf json 
with open("./openai-conf.json") as f:
    api_conf = json.load(f)

# instantiate openai 
client = OpenAI(api_key=api_conf["api_key"])

# First of we define our Python custom functions
def compute_mean(inp_values: list) -> float:
    return round(np.mean(inp_values), 4)

def compute_median(inp_values: list) -> float:
    return round(np.median(inp_values), 4)

def compute_quantile(inp_values: list, q: float) -> float:
    return round(np.quantile(inp_values, q), 4)

# Define a "new" custom functions to check whether the gpt is actually calling a custom function
def compute_chi(a: float) -> float:
    return (a**2+1) - a + 2

# Create a dictionary of available functions 
available_function_dict = {
     "compute_mean": compute_mean,
     "compute_median": compute_median,
     "compute_quantile": compute_quantile,
     "compute_chi": compute_chi
    }

# Now we need to describe the functions to GPT as follows:
compute_mean_desc = {
    "name": "compute_mean",
    "description": "Compute the mean of a provided list of values. Call this function whenever you need to compute the mean of numbers, for example when a user asks 'Compute the mean of these numbers: 1, 2, 3, 4'",
    "parameters": {
        "type": "object",
        "properties": {
            "inp_values": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The input list of values"
            }
        },
        "required": ["inp_values"],
        "additionalProperties": False
    }
}

compute_median_desc = {
    "name": "compute_mean",
    "description": "Compute the median of a provided list of values. Call this function whenever you need to compute the median of numbers, for example when a user asks 'Compute the median of these numbers: 1, 2, 3, 4'",
    "parameters": {
        "type": "object",
        "properties": {
            "inp_values": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The input list of values"
            }
        },
        "required": ["inp_values"],
        "additionalProperties": False
    }
}

compute_quantile_desc = {
    "name": "compute_quantile",
    "description": "Compute the q-quantile of a provided list of values and with q specified by the user. Call this function whenever you need to compute the q-quantile of numbers, for example when a user asks 'Compute the 0.95-quantile of these numbers: 1, 2, 3, 4'",
    "parameters": {
        "type": "object",
        "properties": {
            "inp_values": {
                "type": "array",
                "items": {
                    "type": "number"
                },
                "description": "The input list of values"
            }, 
            "q": {
                "type": "number",
                "description": "The qth quantile to compute"
            }
        },
        "required": ["inp_values", "q"],
        "additionalProperties": False
    }
}

compute_chi_desc = {
    "name": "compute_chi",
    "description": "Compute a metric called chi of an input number. Call this function whenever you need to compute the metric chi of a number, for example when a user asks 'Compute the chi of 3.5'",
    "parameters": {
        "type": "object",
        "properties": {
            "a": {
                "type": "number",
                "description": "The input number"
            }
        },
        "required": ["a"],
        "additionalProperties": False
    }
}

custom_functions = [compute_mean_desc, compute_median_desc, compute_quantile_desc, compute_chi_desc]

# Now we code the while-cycle allowing to interact with GPT as a ChatBot, passing
# required arguments
def chatbot(available_function_dict, gpt_custom_functions):
  # Create a list to store all the messages for context
  messages = [
    {"role": "system", 
    "content": """You are a helpful support assistant. Use the supplied tools and functions to assist the user. Follow these rules:
    
    - before calling any custom functions, always ask the user to confirm the provided input values
    - if the user asks you to compute something you don't have a custom function for, reply with 'I cannot compute this metric, sorry'
    """},
  ]

  # Keep repeating the following
  while True:
    # Prompt user for input
    message = input("User: ")

    # Exit program if user inputs "quit"
    if message.lower() == "quit":
      break

    # Add each new message to the list
    messages.append({"role": "user", "content": message})

    # Request gpt-4o for chat completion, passing gpt_custom_functions to functions argument
    # function_call = 'auto' allows giving gpt the freedom to choose whether to call or not a given function
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        functions=gpt_custom_functions,
        function_call = "auto"
    )

    response_message = response.choices[0].message

    # Check if a function calling has been performed
    if dict(response_message).get('function_call'):
        
        # Identify which function call was invoked
        function_called = response_message.function_call.name
        
        # Extracting the arguments
        function_args  = json.loads(response_message.function_call.arguments)
        
        fuction_to_call = available_function_dict[function_called]
        # Finally call the Python function with inputs provided by the user
        response_message = fuction_to_call(*list(function_args .values()))
        
    else:
        # Otherwise provide the model response 
        response_message = response_message.content

    # Print the response and add it to the messages list
    chat_message = response_message
    print(f"Bot: {chat_message}")
    
    # Appending the conversation to current messages so that the model has the conversation "memory"
    messages.append({"role": "assistant", "content": str(chat_message)})

if __name__ == "__main__":
  print("Start chatting with the bot (type 'quit' to stop)!")
  chatbot(available_function_dict, custom_functions)

