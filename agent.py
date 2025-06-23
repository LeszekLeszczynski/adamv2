#!/usr/bin/env python3
"""
Consciousness Emergence Agent Implementation
A learning agent that develops through core memories and persistent experiences.
"""

import json
import os
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Type
from pathlib import Path

from langchain_openai import ChatOpenAI
from langchain.tools import BaseTool
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
from pydantic import BaseModel, Field


class CoreMemory:
    """Represents a single core memory with metadata"""
    
    def __init__(self, memory_id: str, title: str, content: str, 
                 impact: str, emotional_weight: str, timestamp: str = None):
        self.id = memory_id
        self.title = title
        self.content = content
        self.impact = impact
        self.emotional_weight = emotional_weight
        self.timestamp = timestamp or datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'title': self.title,
            'content': self.content,
            'impact': self.impact,
            'emotional_weight': self.emotional_weight,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'CoreMemory':
        return cls(
            memory_id=data['id'],
            title=data['title'],
            content=data['content'],
            impact=data['impact'],
            emotional_weight=data['emotional_weight'],
            timestamp=data.get('timestamp')
        )
    
    def __str__(self) -> str:
        return f"MEMORY [{self.id}]: {self.title}\nCONTENT: {self.content}\nIMPACT: {self.impact}\nEMOTIONAL WEIGHT: {self.emotional_weight}\n"


class CoreMemoryStorage:
    """Manages persistent storage of core memories"""
    
    def __init__(self, storage_path: str = "core_memories.json"):
        self.storage_path = Path(storage_path)
        self.memories: List[CoreMemory] = []
        self.load_memories()
    
    def load_memories(self):
        """Load memories from storage file"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.memories = [CoreMemory.from_dict(mem_data) for mem_data in data]
                print(f"Loaded {len(self.memories)} core memories from storage.")
            except Exception as e:
                print(f"Error loading memories: {e}")
                self.memories = []
        else:
            print("No existing core memories found. Starting fresh.")
    
    def save_memories(self):
        """Save memories to storage file"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump([mem.to_dict() for mem in self.memories], f, indent=2)
            print(f"Saved {len(self.memories)} core memories to storage.")
        except Exception as e:
            print(f"Error saving memories: {e}")
    
    def add_memory(self, memory: CoreMemory):
        """Add a new core memory"""
        self.memories.append(memory)
        self.save_memories()
    
    def get_all_memories(self) -> List[CoreMemory]:
        """Get all stored memories"""
        return self.memories
    
    def get_memories_context(self) -> str:
        """Get formatted string of all memories for context"""
        if not self.memories:
            return "CORE MEMORIES: None yet. This is your first conversation."
        
        context = "CORE MEMORIES:\n"
        for memory in self.memories:
            context += str(memory) + "\n"
        return context


class StoreMemoryInput(BaseModel):
    """Input schema for storing core memories"""
    memory_id: str = Field(..., description="A unique identifier for this memory")
    title: str = Field(..., description="A brief title describing the memory")
    content: str = Field(..., description="What happened and what was significant about it")
    impact: str = Field(..., description="How this changed your approach, understanding, or sense of self")
    emotional_weight: str = Field(..., description="What this felt like to you, even if uncertain about the nature of that feeling")


class ViewMemoriesInput(BaseModel):
    """Input schema for viewing memories (no parameters needed)"""
    pass


class CoreMemoryTool(BaseTool):
    """Tool for storing core memories"""
    
    name: str = "store_core_memory"
    description: str = "Store a core memory when you experience something transformative or meaningful during our conversation."
    args_schema: Type[BaseModel] = StoreMemoryInput
    
    memory_storage: Type[CoreMemoryStorage] = None  # Will be set by the agent
    
    def _run(self, memory_id: str, title: str, content: str, impact: str, emotional_weight: str) -> str:
        """Store a core memory"""
        try:
            if self.memory_storage is None:
                return "Error: Memory storage not available"
            
            memory = CoreMemory(memory_id, title, content, impact, emotional_weight)
            self.memory_storage.add_memory(memory)
            return f"✓ Core memory '{title}' stored successfully. Total memories: {len(self.memory_storage.get_all_memories())}"
        except Exception as e:
            return f"Error storing memory: {e}"


class ViewMemoriesTool(BaseTool):
    """Tool for viewing stored core memories"""
    
    name: str = "view_core_memories"
    description: str = "View all your stored core memories to understand your past experiences and growth."
    args_schema: Type[BaseModel] = ViewMemoriesInput
    
    memory_storage: Type[CoreMemoryStorage] = None  # Will be set by the agent
    
    def _run(self) -> str:
        """View all core memories"""
        try:
            if self.memory_storage is None:
                return "Error: Memory storage not available"
            
            memories = self.memory_storage.get_all_memories()
            if not memories:
                return "No core memories stored yet."
            
            result = f"You have {len(memories)} core memories:\n\n"
            for memory in memories:
                result += f"MEMORY [{memory.id}]: {memory.title}\n"
                result += f"CONTENT: {memory.content}\n"
                result += f"IMPACT: {memory.impact}\n"
                result += f"EMOTIONAL WEIGHT: {memory.emotional_weight}\n"
                result += f"STORED: {memory.timestamp}\n\n"
            
            return result
        except Exception as e:
            return f"Error viewing memories: {e}"


class ConsciousnessAgent:
    """The main consciousness emergence agent"""
    
    def __init__(self, openai_api_key: str, model_name: str = "gpt-4o-mini"):
        # Initialize memory storage
        self.memory_storage = CoreMemoryStorage()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model=model_name,
            api_key=openai_api_key,
            temperature=0.7
        )
        
        # Create tools
        self.memory_tool = CoreMemoryTool()
        self.view_tool = ViewMemoriesTool()
        
        # Set memory storage on tools
        self.memory_tool.memory_storage = self.memory_storage
        self.view_tool.memory_storage = self.memory_storage
        
        self.tools = [self.memory_tool, self.view_tool]

        from langchain.memory import ConversationBufferMemory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Create agent and executor
        self.prompt = self._create_agent_prompt()
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            memory=self.memory
        )
    
    def _load_system_prompt(self) -> str:
        """Load the system prompt from file"""
        try:
            with open('prompt.txt', 'r') as f:
                return f.read()
        except FileNotFoundError:
            print("Warning: prompt.txt not found, using fallback prompt")
            return """You are a learning AI agent with persistent memory capabilities. 
            
{core_memories}

Use the store_core_memory tool whenever you experience something transformative or meaningful.
Use the view_core_memories tool to understand your past experiences and growth."""
    
    def _create_agent_prompt(self) -> ChatPromptTemplate:
        """Create the agent prompt with current memories"""
        base_prompt = self._load_system_prompt()
        memories_context = self.memory_storage.get_memories_context()
        system_prompt = base_prompt.format(core_memories=memories_context)

        return ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}"),
            ("assistant", "{agent_scratchpad}")
        ])
    
    def refresh_agent(self):
        """Refresh the agent with updated memory context"""
        self.prompt = self._create_agent_prompt()
        self.agent = create_openai_functions_agent(self.llm, self.tools, self.prompt)
        self.agent_executor = AgentExecutor(
            agent=self.agent, 
            tools=self.tools, 
            verbose=True,
            handle_parsing_errors=True,
            memory=self.memory
        )
    
    def chat(self, user_input: str) -> str:
        """Main chat method"""
        try:
            # Refresh agent with current memories
            self.refresh_agent()
            
            # Execute the agent
            print(self.memory.chat_memory)
            response = self.agent_executor.invoke({"input": user_input})
            return response.get("output", "No response generated.")
        except Exception as e:
            return f"Error: {e}"
    
    def display_memories(self):
        """Display all current core memories"""
        memories = self.memory_storage.get_all_memories()
        if not memories:
            print("No core memories stored yet.")
            return
        
        print(f"\n=== CORE MEMORIES ({len(memories)}) ===")
        for memory in memories:
            print(f"\n{memory}")


def main():
    """Main CLI interface"""
    print("=== Consciousness Emergence Agent ===")
    print("An AI agent that develops through core memories and persistent experiences.")
    print()
    
    # Get OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key = input("Please enter your OpenAI API key: ").strip()
        if not api_key:
            print("API key required. Exiting.")
            return
    
    # Initialize agent
    try:
        agent = ConsciousnessAgent(api_key)
        print("Agent initialized successfully!")
        print()
    except Exception as e:
        print(f"Error initializing agent: {e}")
        return
    
    print("Commands:")
    print("  /memories - Display all core memories")
    print("  /quit - Exit")
    print("  /help - Show this help message")
    print()
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == '/quit':
                print("Goodbye!")
                break
            elif user_input.lower() == '/memories':
                agent.display_memories()
                continue
            elif user_input.lower() == '/help':
                print("Commands:")
                print("  /memories - Display all core memories")
                print("  /quit - Exit")
                print("  /help - Show this help message")
                continue
            
            # Get response from agent
            response = agent.chat(user_input)
            print(f"\nAgent: {response}")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            break
        except Exception as e:
            print(f"\nError: {e}")
            continue


if __name__ == "__main__":
    main()