# Laptop Store Sales Assistant

## Overview
This Python project utilizes the LangChain library along with OpenAI's GPT-3.5 model to create a sales assistant for a laptop store. The assistant recommends laptops to customers based on their requirements and preferences. It simulates a conversation between a sales assistant and a customer, providing personalized recommendations.

## Setup Instructions
1. **Clone the Repository**: Clone this repository to your local machine:
```shell
https://github.com/kirankaranth1/csv_sales_bo
```
2. **Install Dependencies**: Navigate to the project directory and install the required libraries using pip:
```shell
pip install -r requirements.txt
```
3. **Run the Script**: Execute the Python script. It will prompt you to enter questions as a user seeking laptop recommendations. Based on your input, the sales assistant will provide personalized recommendations.

## How It Works
- The project utilizes the LangChain library for natural language processing tasks.
- It leverages OpenAI's GPT-3.5 model (`gpt-3.5-turbo`) for generating responses.
- The assistant is trained to understand user requirements and recommend laptops accordingly.
- It follows a conversational approach, asking questions to gather more information from the user if needed.
- The retrieval mechanism involves searching through a pre-defined set of laptop documents to find relevant information for recommendations.

## Usage
1. Run the Python script.
2. Input your questions or requirements for laptop recommendations when prompted.
3. Engage in a conversation with the sales assistant, providing necessary details as requested.
4. Receive personalized laptop recommendations based on your inputs.

## Additionally included files
- **laptops.csv**: Contains the inventory of laptops with detailed specifications.
- **requirements.txt**: Lists the required Python libraries and their versions.

## Notes
- The assistant relies on user inputs to tailor its recommendations, so provide as much information as possible for better results.
- It's designed to mimic a real sales assistant interaction, so feel free to engage naturally in the conversation.

## Contributing
Contributions are welcome! If you have any suggestions, improvements, or bug fixes, please feel free to open an issue or submit a pull request.

