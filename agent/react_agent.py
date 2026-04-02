from langchain.agents import create_agent
from model.factory import chat_model
from utils.prompt_loader import load_system_prompt
from agent.tools.agent_tools import rag_summarize, get_weather, get_user_location, get_user_id, get_current_month, fill_context_for_report, fetch_external_data
from agent.tools.middleware import get_registered_middlewares, MiddlewareRegistrationError
from utils.logger_handler import logger

class ReactAgent:
    def __init__(self):
        try:
            self.agent = create_agent(
                model = chat_model,
                system_prompt = load_system_prompt(),
                tools = [
                    rag_summarize,
                    get_weather,
                    get_user_location,
                    get_user_id,
                    get_current_month,
                    fill_context_for_report,
                    fetch_external_data
                ],
                middleware = get_registered_middlewares()
            )
        except MiddlewareRegistrationError as e:
            logger.error(str(e))
            raise
        
    def excute_stream(self, query: str):
        input_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": query
                    }
            ]
        }
        
        for chunk in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chunk["messages"][-1]
            if latest_message.content:
                yield latest_message.content.strip() + "\n"
                
