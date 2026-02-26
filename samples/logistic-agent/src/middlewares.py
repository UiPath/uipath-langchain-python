from langchain.agents.middleware import AgentMiddleware

class DisableParallelToolCallsMiddleware(AgentMiddleware):
    
    def wrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return handler(request)
    
    async def awrap_model_call(self, request, handler):
        request.model_settings["parallel_tool_calls"] = False
        return await handler(request)